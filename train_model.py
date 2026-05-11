import os
import json
import gc
import torch
import torch.nn.functional as F
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, LinearLR, SequentialLR
from transformers import (
    Qwen3VLForConditionalGeneration,
    AutoProcessor,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, TaskType
from tqdm import tqdm
import warnings

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["TOKENIZERS_PARALLELISM"]  = "false"

warnings.filterwarnings("ignore", category=UserWarning, module="peft")
warnings.filterwarnings("ignore", category=UserWarning, module="transformers")

PROJECT_ROOT  = Path(__file__).parent
DATASET_DIR   = PROJECT_ROOT / "dataset"
MODELS_DIR    = PROJECT_ROOT / "models"
PROMPTS_DIR   = PROJECT_ROOT / "prompts"
ADAPTERS_DIR  = MODELS_DIR / "adapters"
MAX_IMAGE_DIM = 1024

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

TASKS = {
    "answer":  {"name": "qa_adapter",   "dataset_key": "answer",  "grad_acc": 4, "distill_mode": "word_kd"},
    "dsl_gen": {"name": "dsl_adapter",  "dataset_key": "dsl_gen", "grad_acc": 2, "distill_mode": "seq_kd"},
    "edit":    {"name": "edit_adapter", "dataset_key": "edit",    "grad_acc": 2, "distill_mode": "seq_kd"},
}

ALPHA_KL     = 0.3
DISTILL_TEMP = 1.5

LR              = 2e-4
EPOCHS_PER_TASK = 2
BATCH_SIZE      = 1
LORA_R          = 8
LORA_ALPHA      = 16
WARMUP_RATIO    = 0.05
MAX_SEQ_LEN     = 2048

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32       = True
torch.backends.cudnn.benchmark        = False
if hasattr(torch, "set_float32_matmul_precision"):
    torch.set_float32_matmul_precision("high")


class DistillDataset(Dataset):
    def __init__(self, meta_path: Path, processor, task_key: str, distill_mode: str):
        self.meta         = json.loads(meta_path.read_text(encoding="utf-8"))
        self.processor    = processor
        self.task_key     = task_key
        self.distill_mode = distill_mode
        self.eos_id       = processor.tokenizer.eos_token_id
        self.samples      = []

        raw_samples = []
        for entry in self.meta:
            if task_key not in entry.get("steps_completed", []):
                continue

            idx      = entry["index"]
            img_path = None if task_key == "edit" else (PROMPTS_DIR / "img" / entry["image"])

            if task_key == "answer":
                q_text      = (PROMPTS_DIR / "txt" / f"{idx}a.txt").read_text(encoding="utf-8").strip()
                prompt      = f"{q_text}\n{QUESTION_SUFFIX}"
                target_path = DATASET_DIR / "txt"    / f"{idx}.txt"
                logits_path = DATASET_DIR / "logits" / f"{idx}_answer.pt"
            elif task_key == "dsl_gen":
                prompt      = UGDSL_PROMPT
                target_path = DATASET_DIR / "ugdsl"  / f"{idx}.dsl"
                logits_path = DATASET_DIR / "logits" / f"{idx}_dsl.pt"
            else:
                dsl_text    = (DATASET_DIR / "ugdsl" / f"{idx}.dsl").read_text(encoding="utf-8").strip()
                e_text      = (PROMPTS_DIR / "txt" / f"{idx}e.txt").read_text(encoding="utf-8").strip()
                prompt      = f"{dsl_text}\n\n{e_text}\n{EDIT_UGDSL_SUFFIX}"
                target_path = DATASET_DIR / "ugdsl"  / f"{idx}_edited.dsl"
                logits_path = None

            raw_samples.append({
                "img_path":    str(img_path) if img_path else None,
                "prompt":      prompt,
                "target_text": target_path.read_text(encoding="utf-8").strip(),
                "logits_path": str(logits_path) if logits_path else None,
                "idx":         idx,
            })

        if distill_mode == "word_kd":
            print(f"  Pre-loading {len(raw_samples)} logit files into RAM...")
            for s in raw_samples:
                topk_ids, topk_vals = self._parse_logits(s["logits_path"])
                self.samples.append({**s, "topk_ids": topk_ids, "topk_vals": topk_vals})
        else:
            self.samples = [{**s, "topk_ids": None, "topk_vals": None} for s in raw_samples]

        print(f"  Dataset ready: {len(self.samples)} valid samples.")

    @staticmethod
    def _parse_logits(path: str):
        t_data = torch.load(path, map_location="cpu", weights_only=False)
        ids_list, vals_list = [], []
        for step in t_data["logits_list"]:
            ids  = torch.as_tensor(step.get("top_k_ids",    []), dtype=torch.long)
            vals = torch.as_tensor(step.get("top_k_logits", step.get("top_k_vals", [])), dtype=torch.float32)
            ids_list.append(ids)
            vals_list.append(vals)
        return torch.stack(ids_list, dim=0), torch.stack(vals_list, dim=0)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        s   = self.samples[i]
        eos = self.eos_id

        messages = [{"role": "user", "content": []}]
        img = None

        if s["img_path"] is not None:
            img = Image.open(s["img_path"]).convert("RGB")
            w, h = img.size
            scale = MAX_IMAGE_DIM / max(w, h)
            if scale < 1.0:
                img = img.resize((int(w * scale), int(h * scale)), Image.Resampling.LANCZOS)
            messages[0]["content"].append({"type": "image"})

        messages[0]["content"].append({"type": "text", "text": s["prompt"]})

        text   = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=text, images=[img] if img else None, return_tensors="pt")

        img_tensor = inputs.get("pixel_values")   if img else None
        grid_thw   = inputs.get("image_grid_thw") if img else None

        target_tokens = self.processor.tokenizer(
            s["target_text"], return_tensors="pt", add_special_tokens=False
        )
        eos_tensor = torch.tensor([[eos]], dtype=torch.long)
        full_ids   = torch.cat(
            [inputs["input_ids"], target_tokens["input_ids"], eos_tensor], dim=1
        )

        labels = full_ids.clone()
        labels[0, : inputs["input_ids"].shape[1]] = -100

        if full_ids.shape[1] > MAX_SEQ_LEN:
            full_ids = full_ids[:, :MAX_SEQ_LEN]
            labels   = labels[:, :MAX_SEQ_LEN]

        return {
            "input_ids":         full_ids.squeeze(0),
            "attention_mask":    torch.ones(full_ids.shape[1], dtype=torch.long),
            "pixel_values":      img_tensor,
            "image_grid_thw":    grid_thw,
            "labels":            labels.squeeze(0),
            "teacher_topk_ids":  s["topk_ids"],
            "teacher_topk_vals": s["topk_vals"],
            "prompt_len":        inputs["input_ids"].shape[1],
            "idx":               s["idx"],
        }


def collate_fn(batch):
    input_ids      = torch.nn.utils.rnn.pad_sequence(
        [x["input_ids"]      for x in batch], batch_first=True, padding_value=0
    )
    attention_mask = torch.nn.utils.rnn.pad_sequence(
        [x["attention_mask"] for x in batch], batch_first=True, padding_value=0
    )
    labels         = torch.nn.utils.rnn.pad_sequence(
        [x["labels"]         for x in batch], batch_first=True, padding_value=-100
    )

    pixel_values   = (
        torch.cat([x["pixel_values"]   for x in batch], dim=0)
        if batch[0]["pixel_values"] is not None else None
    )
    image_grid_thw = (
        torch.cat([x["image_grid_thw"] for x in batch], dim=0)
        if batch[0]["image_grid_thw"] is not None else None
    )

    return {
        "input_ids":         input_ids,
        "attention_mask":    attention_mask,
        "pixel_values":      pixel_values,
        "image_grid_thw":    image_grid_thw,
        "labels":            labels,
        "teacher_topk_ids":  [x["teacher_topk_ids"]  for x in batch],
        "teacher_topk_vals": [x["teacher_topk_vals"] for x in batch],
        "prompt_lens":       [x["prompt_len"]         for x in batch],
    }


def compute_seq_kd_loss(student_logits, labels) -> torch.Tensor:
    shift_logits = student_logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    return F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=-100,
    )


def compute_word_kd_loss(
    student_logits,
    labels,
    teacher_topk_ids_list,
    teacher_topk_vals_list,
    prompt_lens,
    alpha: float = ALPHA_KL,
    temp:  float = DISTILL_TEMP,
) -> torch.Tensor:

    shift_logits = student_logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()

    ce_loss = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=-100,
    )

    kl_loss      = torch.tensor(0.0, device=student_logits.device)
    valid_tokens = 0.0
    device       = student_logits.device

    for b in range(len(prompt_lens)):
        t_ids  = teacher_topk_ids_list[b].to(device)
        t_vals = teacher_topk_vals_list[b].to(device)

        prompt_offset = prompt_lens[b] - 1
        teacher_len   = t_ids.shape[0]
        available     = shift_logits.shape[1] - prompt_offset
        match_len     = min(available, teacher_len)

        s_logits_gen = shift_logits[b, prompt_offset : prompt_offset + match_len, :]
        t_ids_m      = t_ids[:match_len]
        t_vals_m     = t_vals[:match_len]

        valid_mask = shift_labels[b, prompt_offset : prompt_offset + match_len] != -100

        log_Z            = torch.logsumexp(s_logits_gen / temp, dim=-1, keepdim=True)
        s_log_probs_at_k = s_logits_gen.gather(-1, t_ids_m) / temp - log_Z
        t_probs_k        = F.softmax(t_vals_m / temp, dim=-1)

        kl = (t_probs_k * (t_probs_k.log() - s_log_probs_at_k)).sum(-1)

        kl_valid      = kl[valid_mask]
        kl_loss      += kl_valid.sum()
        valid_tokens += kl_valid.numel()

    kl_loss = (kl_loss / valid_tokens) * (temp ** 2)

    return (1.0 - alpha) * ce_loss + alpha * kl_loss


def train_task(task_key, task_cfg, model, processor, optimizer, scheduler,
               dataloader, device, grad_acc_steps):
    model.train()
    total_steps  = len(dataloader) * EPOCHS_PER_TASK
    global_step  = 0
    distill_mode = task_cfg["distill_mode"]
    optimizer.zero_grad()

    for epoch in range(EPOCHS_PER_TASK):
        epoch_losses = []
        pbar = tqdm(
            dataloader,
            desc=f"[{task_cfg['name']} / {distill_mode}] Epoch {epoch + 1}/{EPOCHS_PER_TASK}"
        )

        for batch in pbar:
            batch = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }

            outputs = model(
                input_ids      = batch["input_ids"],
                attention_mask = batch["attention_mask"],
                pixel_values   = batch["pixel_values"],
                image_grid_thw = batch.get("image_grid_thw"),
                labels         = batch["labels"],
                use_cache      = False,
            )

            logits = outputs.logits
            del outputs

            if distill_mode == "seq_kd":
                loss = compute_seq_kd_loss(logits, batch["labels"])
            else:
                loss = compute_word_kd_loss(
                    logits,
                    batch["labels"],
                    batch["teacher_topk_ids"],
                    batch["teacher_topk_vals"],
                    batch["prompt_lens"],
                )

            loss = loss / grad_acc_steps
            loss.backward()

            current_loss = loss.item() * grad_acc_steps
            epoch_losses.append(current_loss)

            if (global_step + 1) % grad_acc_steps == 0 or global_step == total_steps - 1:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            pbar.set_postfix(
                loss = f"{current_loss:.4f}",
                lr   = f"{scheduler.get_last_lr()[0]:.1e}",
            )
            global_step += 1

        print(
            f"\n[{task_cfg['name']}] Epoch {epoch + 1} → "
            f"Mean: {sum(epoch_losses)/len(epoch_losses):.4f} | "
            f"Min: {min(epoch_losses):.4f} | Max: {max(epoch_losses):.4f} | "
        )
        torch.cuda.reset_peak_memory_stats()

    return model


def make_lora_config() -> LoraConfig:
    return LoraConfig(
        task_type      = TaskType.CAUSAL_LM,
        r              = LORA_R,
        lora_alpha     = LORA_ALPHA,
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                          "gate_proj", "up_proj", "down_proj"],
        lora_dropout   = 0.05,
        bias           = "none",
        init_lora_weights = True,
    )


def load_base_model(student_path: Path, bnb_config: BitsAndBytesConfig):
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        student_path,
        quantization_config = bnb_config,
        torch_dtype         = torch.bfloat16,
        device_map          = "cuda:0",
        attn_implementation = "sdpa",
        trust_remote_code   = True,
        low_cpu_mem_usage   = True,
    )
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    return model


def main():
    print("Инициализация QLoRA окружения (RTX 5070 12 GB)...")
    device = torch.device("cuda:0")

    student_path = MODELS_DIR / "analyzer"
    if not student_path.exists():
        raise FileNotFoundError(f"Базовая модель не найдена: {student_path}")

    ADAPTERS_DIR.mkdir(parents=True, exist_ok=True)
    meta_path = DATASET_DIR / "dataset_meta.json"

    processor = AutoProcessor.from_pretrained(student_path, trust_remote_code=True)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit              = True,
        bnb_4bit_compute_dtype    = torch.bfloat16,
        bnb_4bit_quant_type       = "nf4",
        bnb_4bit_use_double_quant = True,
        bnb_4bit_quant_storage    = torch.bfloat16,
    )

    for task_key, task_cfg in TASKS.items():
        adapter_name = task_cfg["name"]
        save_path    = ADAPTERS_DIR / adapter_name
        distill_mode = task_cfg["distill_mode"]

        print(f"\n{'='*60}")
        print(f"Адаптер : {adapter_name}")
        print(f"Режим   : {distill_mode.upper()}")
        print(f"{'='*60}")

        dataset    = DistillDataset(meta_path, processor, task_cfg["dataset_key"], distill_mode)
        dataloader = DataLoader(
            dataset, batch_size=BATCH_SIZE, collate_fn=collate_fn,
            shuffle=True, num_workers=0, pin_memory=True,
        )

        print("Загрузка базовой модели...")
        model = load_base_model(student_path, bnb_config)
        model = get_peft_model(model, make_lora_config(), adapter_name=adapter_name)
        model.set_adapter(adapter_name)
        model.print_trainable_parameters()

        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()), 
            lr=LR,
            weight_decay=1e-4
        )

        steps_per_epoch = len(dataloader)
        total_steps = steps_per_epoch * EPOCHS_PER_TASK
        warmup_steps = max(1, int(total_steps * WARMUP_RATIO)) 

        cosine_restart = CosineAnnealingWarmRestarts(
            optimizer,
            T_0=steps_per_epoch,
            T_mult=1
            eta_min=1e-6,
            last_epoch=-1
        )

        warmup_scheduler = LinearLR(
            optimizer,
            start_factor=1e-4,
            end_factor=1.0,
            total_iters=warmup_steps
        )

        scheduler = SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, cosine_restart],
            milestones=[warmup_steps]
        )

        model = train_task(
            task_key, task_cfg, model, processor,
            optimizer, scheduler, dataloader, device,
            grad_acc_steps=task_cfg["grad_acc"],
        )

        model.save_pretrained(str(save_path), selected_adapters=[adapter_name])
        print(f"Адаптер '{adapter_name}' сохранён в {save_path}")

        del model
        del optimizer
        del scheduler
        torch.cuda.empty_cache()
        gc.collect()
        print("GPU память освобождена.\n")

    print("\nДистилляция завершена. Все адаптеры сохранены.")

if __name__ == "__main__":
    main()