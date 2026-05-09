import gc
import json
import re
import time
import os
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, Future
import torch
from PIL import Image
from tqdm import tqdm
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
 
PROJECT_ROOT = Path(__file__).parent
MODELS_DIR   = PROJECT_ROOT / "models"
PROMPTS_DIR  = PROJECT_ROOT / "prompts"
DATASET_DIR  = PROJECT_ROOT / "dataset"
MAX_IMAGE_DIM = 1024
 
torch.backends.cuda.enable_mem_efficient_sdp(True)
torch.set_float32_matmul_precision("high")
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32       = True
torch.backends.cudnn.benchmark        = False

GENERATION_KWARGS = {
    "qa":   {"max_new_tokens": 512,  "temperature": 0.2, "top_p": 0.9, "top_k": 0, "repetition_penalty": 1.1, "do_sample": True},
    "dsl":  {"max_new_tokens": 728,  "repetition_penalty": 1.15, "do_sample": False},
    "edit": {"max_new_tokens": 728,  "repetition_penalty": 1.15, "do_sample": False},
}
 
QUESTION_SUFFIX = """ВЫВОД: Только финальный ответ.
- БЕЗ рассуждений, пояснений, списков или маркеров.
- Число -> просто число. Список -> элементы через запятую.
- ЗАВЕРШИ вывод сразу после ответа."""
 
UGDSL_PROMPT = """Ты - автоматический парсер. Преобразуй ПРИЛОЖЕННОЕ ИЗОБРАЖЕНИЕ в строгий текстовый формат DSL.
 
[ФОРМАТ]
N <id> <label> <x> <y> <node_shape> <color>
E <from> <to> <label>
U <from> <to> <label>
 
[ЧЁТКИЕ ОПРЕДЕЛЕНИЯ]
    N <id> <label> <node_type> <color> - параметры узла (ровно 6 полей после символа N):
        Формы узлов (5-е поле): E=ellipse, D=diamond, I=parallelogram, B=box, C=circle
        Цвета узлов (6-е поле): R=red, G=green, B=blue, Y=yellow, O=orange, P=purple, K=black, W=white, X=gray
        ВАЖНО: Цвет определяй СТРОГО по пикселям изображения. Если не видно -> W. Не угадывай!
        ВАЖНО: Если не видно форму -> E. Не угадывай!
    E / U - параметры связи (ровно 3 поля после E или U):
        E = направленная связь (стрелка)
        U = ненаправленная связь (линия)
        label = текст на связи. Если текста нет -> ставь символ "-"
 
[ПРАВИЛА ГЕНЕРАЦИИ]
1. Строка может начинаться ТОЛЬКО на N, E или U.
2. id: A, B, C... (латиница, БЕЗ РУССКИХ символов).
3. label узлов: UPPER_SNAKE_CASE для текста, цифры для чисел. Пробелов быть НЕ ДОЛЖНО. Пробелы на изображении -> _ в label. Язык оригинала. Без кавычек.
4. x,y: 0–1000. Ось X -> вправо, Y -> вниз. Сохраняй относительную сетку оригинала.
5. Разделитель полей: ровно 1 пробел. Никаких комментариев или выравнивания.
6. ВАЖНО: НЕЛЬЗЯ выводить связи с несуществующими узлами (чьи id не существуют на вторых позициях в строках, начинающихся с N).
7. Между двумя узлами может быть не более одной связи.
 
[ПРИМЕР ВЫВОДА]
N A НАЧАЛО 500 100 E W
N B ВЫКОПАТЬ_ЯМУ 500 200 P B
N C 12 500 300 E W
E A B ВЫКОПАТЬ
U B C -
 
ВЫВОДИ ТОЛЬКО КОД. НАЧИНАЙ С N."""
 
EDIT_UGDSL_SUFFIX = """ЗАДАЧА: Обнови DSL код выше согласно запросу.
 
ПРАВИЛА ИЗМЕНЕНИЯ:
1. Внеси ТОЛЬКО указанное изменение. Остальное скопируй без изменений.
2. Строго соблюдай формат: N <id> <label> <x> <y> <type> <color>, E/U <from> <to> <label>.
3. id: A, B, C... (латиница, БЕЗ РУССКИХ символов).
4. label узлов: UPPER_SNAKE_CASE (пробелы -> _ ) для текста и цифы для чисел.
5. Формы узлов (5-е поле): E=ellipse, D=diamond, I=parallelogram, B=box, C=circle
6. Цвета узлов (color): R=red, G=green, B=blue, Y=yellow, O=orange, P=purple, K=black, W=white, X=gray
7. При удалении объекта все его связи удаляются, если не указано иначе в запросе.
 
ВЫВОД:
- Выведи обновленный код DSL ЦЕЛИКОМ со всеми изменениями и дополнениями.
- БЕЗ текста до и после кода.
- БЕЗ объяснений, БЕЗ комментариев, БЕЗ размышлений
- Код должен начинаться с N и заканчиваться последней строкой E/U."""
 
_IO_POOL: ThreadPoolExecutor = ThreadPoolExecutor(max_workers=3)
_pending_futures: list[Future] = []
 
def _submit(fn, *args) -> None:
    _pending_futures.append(_IO_POOL.submit(fn, *args))
 
def _flush_pending_writes() -> None:
    for f in _pending_futures:
        exc = f.exception()
        if exc:
            raise RuntimeError(f"Background write failed: {exc}") from exc
    _pending_futures.clear()
 
def _write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
 
def _write_logits(path: Path, logits_data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(logits_data, path.with_suffix(".pt"))
 
def atomic_write_json(path: Path, data: list) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)

def build_eos_ids(tokenizer) -> list[int]:
    eos_ids = set()
    if tokenizer.eos_token_id is not None:
        eos_ids.add(tokenizer.eos_token_id)
    for special in ["<|im_end|>", "<|endoftext|>"]:
        tid = tokenizer.convert_tokens_to_ids(special)
        if tid is not None and tid != tokenizer.unk_token_id:
            eos_ids.add(tid)
    return list(eos_ids)
 
def resize_image(img: Image.Image, max_dim: int = MAX_IMAGE_DIM) -> Image.Image:
    w, h = img.size
    scale = max_dim / max(w, h)
    if scale < 1.0:
        return img.resize((int(w * scale), int(h * scale)), Image.Resampling.LANCZOS)
    return img
 
def clean_output(text: str, is_dsl: bool = False) -> str:
    text = text.strip()
    if not text:
        return ""
    text = re.sub(r"^```(?:\w+)?\s*", "", text, flags=re.MULTILINE).strip()
    text = re.sub(r"\s*```\s*$", "", text).strip()
    if is_dsl and text:
        match = re.search(r"^N\s+[A-Z]+\b", text, flags=re.MULTILINE)
        if match:
            text = text[match.start():]
    return text.strip()
 
def extract_logits(tokenizer, input_ids: torch.Tensor,
                   outputs, eos_ids: set[int], k: int = 50) -> dict:
    if not hasattr(outputs, "logits") or not outputs.logits:
        return {"logits_list": []}
    prompt_len  = input_ids.size(1)
    logits_list = []
    for step, logits in enumerate(outputs.logits):
        token_pos = prompt_len + step
        if token_pos >= outputs.sequences.size(1):
            break
        gen_id = outputs.sequences[0, token_pos].item()
        if gen_id in eos_ids:
            break
        l = logits.squeeze(0).cpu()
        topk_vals, topk_ids = torch.topk(l, k=min(k, l.size(0)), sorted=True)
        logits_list.append({
            "step":              step,
            "generated_token_id": gen_id,
            "top_k_ids":         topk_ids.tolist(),
            "top_k_logits":      topk_vals.tolist(),
            "top_k_tokens":      tokenizer.batch_decode(
                                     topk_ids.unsqueeze(1).tolist(),
                                     skip_special_tokens=True),
            "gen_logit":         l[gen_id].item() if gen_id < l.size(0) else float("-inf"),
        })

        del l
        del topk_vals
        del topk_ids

    return {"logits_list": logits_list}
 
def run_step(model, processor, tokenizer, prompt: str,
             image: Image.Image | None, config: dict,
             output_paths: dict, eos_ids: list[int],
             is_dsl: bool = False) -> dict:
 
    messages = [{"role": "user", "content": []}]
    if image is not None:
        messages[0]["content"].append({"type": "image"})
    messages[0]["content"].append({"type": "text", "text": prompt})
 
    text   = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(
        text=text, images=[image] if image else None, return_tensors="pt"
    ).to(model.device)
 
    gen_config = dict(config)
    if not gen_config.get("do_sample", False):
        for key in ["temperature", "top_p", "top_k"]:
            gen_config.pop(key, None)
 
    with torch.inference_mode():
        t0 = time.perf_counter()
        outputs = model.generate(
            **inputs,
            **gen_config,
            output_logits          = True,
            return_dict_in_generate = True,
            pad_token_id           = tokenizer.pad_token_id,
            eos_token_id           = eos_ids,
        )
        duration = round(time.perf_counter() - t0, 3)
 
    out_ids  = outputs.sequences[0, inputs.input_ids.size(1):]
    raw_text = processor.decode(
        out_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    clean_text = clean_output(raw_text, is_dsl=is_dsl)
 
    _submit(_write_text, output_paths["text"], clean_text)
 
    logits_status = None
    try:
        logits_data = extract_logits(tokenizer, inputs.input_ids, outputs, set(eos_ids), k=50)
        _submit(_write_logits, output_paths["logits"], logits_data)
        logits_status = str(output_paths["logits"].with_suffix(".pt"))
    except Exception as e:
        print(f"Logits error ({output_paths['logits'].name}): {e}")
 
    del outputs
    del inputs
    torch.cuda.empty_cache()
    gc.collect()
 
    return {"text": clean_text, "duration": duration, "logits_path": logits_status}
 
def main():
    torch.cuda.empty_cache()
    gc.collect()
 
    print("Loading Qwen3-VL Teacher (8B)...")
    teacher_path = MODELS_DIR / "teacher"
    processor    = AutoProcessor.from_pretrained(teacher_path, trust_remote_code=True)
    tokenizer    = processor.tokenizer
    eos_ids      = build_eos_ids(tokenizer)
    print(f"   EOS token ids: {eos_ids}")
 
    print("Applying 4-bit NF4 + Double Quantization...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit              = True,
        bnb_4bit_compute_dtype    = torch.bfloat16,
        bnb_4bit_quant_type       = "nf4",
        bnb_4bit_use_double_quant = True,
        bnb_4bit_quant_storage    = torch.bfloat16,
    )
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        teacher_path,
        quantization_config = bnb_config,
        torch_dtype         = torch.bfloat16,
        device_map          = "cuda:0",
        attn_implementation = "sdpa",
        trust_remote_code   = True,
        low_cpu_mem_usage   = True,
    ).eval()
 
    torch.cuda.synchronize()
 
    model.generation_config.pad_token_id = tokenizer.pad_token_id
    model.generation_config.eos_token_id = eos_ids
 
    print(
        f"Model loaded. VRAM: "
        f"{torch.cuda.memory_allocated() / 1024**3:.2f} GB / "
        f"{torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB"
    )
 
    print("Warmup...")
    dummy_inputs = processor(text="warmup", return_tensors="pt").to(model.device)
    with torch.inference_mode():
        model.generate(
            **dummy_inputs, max_new_tokens=5,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=eos_ids,
        )
    torch.cuda.synchronize()
    print("Warmup complete.")
 
    for d in ["txt", "ugdsl", "logits"]:
        (DATASET_DIR / d).mkdir(parents=True, exist_ok=True)
 
    img_dir = PROMPTS_DIR / "img"
    if not img_dir.exists():
        raise FileNotFoundError(f"Missing directory: {img_dir}")
 
    valid_ext   = {".png", ".jpg", ".jpeg", ".webp"}
    image_files = sorted(
        [f for f in img_dir.iterdir()
         if f.suffix.lower() in valid_ext and re.search(r"(\d+)", f.stem)],
        key=lambda f: int(re.search(r"(\d+)", f.stem).group(1)),
    )
    if not image_files:
        print("No valid images found.")
        return
    print(f"Found {len(image_files)} images. Starting generation...")
 
    dataset_log_path = DATASET_DIR / "dataset_meta.json"
    timing_log_path  = DATASET_DIR / "generation_times.json"
 
    def load_log(path: Path) -> list:
        if not path.exists():
            return []
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return data if isinstance(data, list) else []
        except Exception as e:
            print(f"Error reading {path.name}: {e}. Starting fresh.")
            return []
 
    dataset_log       = load_log(dataset_log_path)
    timing_log        = load_log(timing_log_path)
    processed_indices = {str(e["index"]) for e in dataset_log if "index" in e}
    print(f"Already processed: {len(processed_indices)} images.")
 
    CHECKPOINT_INTERVAL = 10
    steps_since_save    = 0
    skipped_count       = 0
 
    for img_path in tqdm(image_files, desc="Processing", unit="img"):
        match = re.search(r"(\d+)", img_path.stem)
        if not match:
            continue
        idx = match.group(1)
 
        if idx in processed_indices:
            skipped_count += 1
            continue
 
        img  = resize_image(Image.open(img_path).convert("RGB"))
        meta = {"index": idx, "image": img_path.name, "steps_completed": []}
        times = {}
 
        q_path = PROMPTS_DIR / "txt" / f"{idx}a.txt"
        q_text = q_path.read_text(encoding="utf-8").strip() if q_path.exists() else ""
        res_q  = run_step(
            model, processor, tokenizer,
            f"{q_text}\n{QUESTION_SUFFIX}" if q_text else QUESTION_SUFFIX.strip(),
            img, GENERATION_KWARGS["qa"],
            {
                "text":   DATASET_DIR / "txt"    / f"{idx}.txt",
                "logits": DATASET_DIR / "logits" / f"{idx}_answer",
            },
            eos_ids, is_dsl=False,
        )
        meta.update({"answer": res_q["text"], "answer_logits": res_q["logits_path"]})
        meta["steps_completed"].append("answer")
        times["answer_s"] = res_q["duration"]
 
        res_dsl = run_step(
            model, processor, tokenizer,
            UGDSL_PROMPT, img, GENERATION_KWARGS["dsl"],
            {
                "text":   DATASET_DIR / "ugdsl"  / f"{idx}.dsl",
                "logits": DATASET_DIR / "logits" / f"{idx}_dsl",
            },
            eos_ids, is_dsl=True,
        )
        meta.update({"dsl_code": res_dsl["text"], "dsl_logits": res_dsl["logits_path"]})
        meta["steps_completed"].append("dsl_gen")
        times["dsl_gen_s"] = res_dsl["duration"]
 
        e_path      = PROMPTS_DIR / "txt" / f"{idx}e.txt"
        e_text      = e_path.read_text(encoding="utf-8").strip() if e_path.exists() else ""
        edit_prompt = (
            f"{res_dsl['text']}\n\n{e_text}\n{EDIT_UGDSL_SUFFIX}"
            if e_text
            else f"{res_dsl['text']}\n\n{EDIT_UGDSL_SUFFIX}"
        )
        res_edit = run_step(
            model, processor, tokenizer,
            edit_prompt, None, GENERATION_KWARGS["edit"],
            {
                "text":   DATASET_DIR / "ugdsl"  / f"{idx}_edited.dsl",
                "logits": DATASET_DIR / "logits" / f"{idx}_edit",
            },
            eos_ids, is_dsl=True,
        )
        meta.update({"edited_dsl": res_edit["text"], "edit_logits": res_edit["logits_path"]})
        meta["steps_completed"].append("edit")
        times["edit_s"] = res_edit["duration"]
 
        meta["step_times"] = times
        dataset_log.append(meta)
        timing_log.append({"index": idx, "image": img_path.name, **times})
        processed_indices.add(idx)
        steps_since_save += 1
 
        is_last = (img_path == image_files[-1])
        if steps_since_save >= CHECKPOINT_INTERVAL or is_last:
            _flush_pending_writes()
            atomic_write_json(dataset_log_path, dataset_log)
            atomic_write_json(timing_log_path,  timing_log)
            steps_since_save = 0
 
        if int(idx) % 10 == 0:
            torch.cuda.empty_cache()
            gc.collect()
 
    _IO_POOL.shutdown(wait=True)
    _flush_pending_writes()
    torch.cuda.empty_cache()
    gc.collect()
 
    if timing_log:
        total = sum(
            t.get("answer_s", 0) + t.get("dsl_gen_s", 0) + t.get("edit_s", 0)
            for t in timing_log
        )
        n = len(timing_log)
        print(
            f"\nTotal: {total:.1f}s | Avg: {total/n:.1f}s/img | "
            f"Processed: {n} | Skipped: {skipped_count}"
        )
    else:
        print("No new images processed.")
    print("DONE.")
 
if __name__ == "__main__":
    main()