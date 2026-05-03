import sys
import os
import warnings

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
torch.backends.cuda.enable_mem_efficient_sdp(True)
torch.set_float32_matmul_precision('high')
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = False

warnings.filterwarnings("ignore", message=".*flop counting will not work.*", category=UserWarning, module="torch.utils.flop_counter")
warnings.filterwarnings("ignore", message=".*_check_is_size.*", category=FutureWarning, module="bitsandbytes")

import gc
import re
import time
from pathlib import Path
from io import BytesIO
import gradio as gr
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
from peft import PeftModel

PROJECT_ROOT = Path(__file__).parent
MODELS_DIR = PROJECT_ROOT / "models"
ADAPTERS_DIR = MODELS_DIR / "adapters_bf16_2B"
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

GENERATION_KWARGS = {
    "qa": {"max_new_tokens": 512, "temperature": 0.2, "top_p": 0.9, "top_k": 0, "repetition_penalty": 1.1, "do_sample": True},
    "dsl": {"max_new_tokens": 728, "repetition_penalty": 1.15, "do_sample": False},
    "edit": {"max_new_tokens": 728, "repetition_penalty": 1.15, "do_sample": False}
}

MODEL = None
PROCESSOR = None
COLOR_MAP = {
    'R': '#E74C3C', 'G': '#2ECC71', 'B': '#3498DB', 'Y': '#F1C40F',
    'O': '#E67E22', 'P': '#9B59B6', 'K': '#2C3E50', 'W': '#ECF0F1',
    'X': '#95A5A6'
}

def resize_image(img, max_dim=MAX_IMAGE_DIM):
    w, h = img.size
    scale = max_dim / max(w, h)
    return img.resize((int(w * scale), int(h * scale)), Image.Resampling.BILINEAR) if scale < 1.0 else img

def clean_output(text, is_dsl=False):
    text = text.strip()
    if not text: return ""
    text = re.sub(r"^```(?:\w+)?\s*", "", text, flags=re.MULTILINE).strip()
    text = re.sub(r"\s*```\s*$", "", text).strip()
    if is_dsl and text:
        match = re.search(r"^N\s+[A-Z]+\b", text, flags=re.MULTILINE)
        if match: text = text[match.start():]
    return text.strip()

def init_model():
    global MODEL, PROCESSOR
    base_path = MODELS_DIR / "analyzer-2B"
    if not base_path.exists():
        raise FileNotFoundError(f"Базовая модель не найдена: {base_path}")
    if not ADAPTERS_DIR.exists():
        raise FileNotFoundError(f"Директория адаптеров не найдена: {ADAPTERS_DIR}")

    print("Загрузка Qwen3-VL-2B (4-bit NF4)...")
    PROCESSOR = AutoProcessor.from_pretrained(base_path, trust_remote_code=True)
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    base_model = Qwen3VLForConditionalGeneration.from_pretrained(
        base_path,
        quantization_config=bnb_config,
        torch_dtype=torch.bfloat16,
        device_map="cuda:0",
        attn_implementation="sdpa",
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    ).eval()

    print("Подгрузка QLoRA адаптеров...")
    adapter_paths = {
        "qa": ADAPTERS_DIR / "qa_adapter",
        "dsl": ADAPTERS_DIR / "dsl_adapter", 
        "edit": ADAPTERS_DIR / "edit_adapter"
    }
    
    for task, path in adapter_paths.items():
        if not (path / "adapter_config.json").exists():
            raise FileNotFoundError(f"Адаптер {task} не найден: {path}")

    MODEL = PeftModel.from_pretrained(base_model, str(adapter_paths["qa"]), adapter_name="qa_adapter")
    MODEL.load_adapter(str(adapter_paths["dsl"]), adapter_name="dsl_adapter")
    MODEL.load_adapter(str(adapter_paths["edit"]), adapter_name="edit_adapter")
    MODEL.eval()
    print(f"Адаптеры готовы: {list(MODEL.peft_config.keys())}")

def run_inference(prompt, image, config, is_dsl=False):
    messages = [{"role": "user", "content": []}]
    if image is not None:
        messages[0]["content"].append({"type": "image"})
    messages[0]["content"].append({"type": "text", "text": prompt})

    text = PROCESSOR.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = PROCESSOR(text=text, images=[image] if image else None, return_tensors="pt").to(MODEL.device)

    gen_config = dict(config)
    if not gen_config.get("do_sample", False):
        for key in ["temperature", "top_p", "top_k"]: gen_config.pop(key, None)

    with torch.inference_mode():
        outputs = MODEL.generate(
            **inputs, **gen_config,
            pad_token_id=PROCESSOR.tokenizer.pad_token_id,
            eos_token_id=PROCESSOR.tokenizer.eos_token_id
        )

    out_ids = outputs[0, inputs.input_ids.size(1):]
    raw_text = PROCESSOR.decode(out_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    clean_text = clean_output(raw_text, is_dsl=is_dsl)

    del outputs, inputs
    torch.cuda.empty_cache()
    return clean_text

def render_dsl(dsl_text):
    if not dsl_text or len(dsl_text.strip()) < 10:
        return None

    nodes = {}
    edges = []
    try:
        for line in dsl_text.strip().split('\n'):
            line = line.strip()
            if not line: continue
            parts = line.split()
            if parts[0] == 'N' and len(parts) >= 7:
                nid, label, x, y, shape, color = parts[1], parts[2], float(parts[3]), float(parts[4]), parts[5], parts[6]
                nodes[nid] = {'label': label, 'x': x, 'y': y, 'shape': shape, 'color': COLOR_MAP.get(color, '#ECF0F1')}
            elif parts[0] in ('E', 'U') and len(parts) >= 4:
                etype, src, dst, label = parts[0], parts[1], parts[2], parts[3]
                edges.append({'type': etype, 'from': src, 'to': dst, 'label': label})
    except Exception:
        return None

    if not nodes:
        return None

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_xlim(-80, 1080)
    ax.set_ylim(1080, -80)
    ax.set_aspect('equal')
    ax.axis('off')

    for nid in nodes:
        node = nodes[nid]
        node['display_label'] = node['label'].replace('_', ' ').title()
        
        words = node['display_label'].split()
        lines, current_line, current_len = [], [], 0
        for w in words:
            if current_len + len(w) + (1 if current_line else 0) > 16:
                lines.append(' '.join(current_line))
                current_line = [w]
                current_len = len(w)
            else:
                current_line.append(w)
                current_len += len(w) + 1
        if current_line:
            lines.append(' '.join(current_line))
            
        node['label_lines'] = lines
        max_line_len = max(len(l) for l in lines) if lines else 0
        num_lines = len(lines)
        
        node['w'] = max(140, max_line_len * 7.5 + 30)
        node['h'] = max(70, num_lines * 18 + 20)
        if node['shape'] == 'C':
            node['r'] = max(60, max(node['w'], node['h']) / 2 + 15)

    for edge in edges:
        src = nodes.get(edge['from'])
        dst = nodes.get(edge['to'])
        if not src or not dst: continue
        x1, y1 = src['x'], src['y']
        x2, y2 = dst['x'], dst['y']

        if edge['type'] == 'E':
            ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                        arrowprops=dict(arrowstyle='->', color='#2C3E50', lw=2.5, shrinkA=5, shrinkB=5))
        else:
            ax.plot([x1, x2], [y1, y2], color='#2C3E50', lw=2.5, zorder=1)

        if edge['label'] and edge['label'] != '-':
            mid_x, mid_y = (x1+x2)/2, (y1+y2)/2
            ax.text(mid_x, mid_y, edge['label'].replace('_', ' ').title(), ha='center', va='center', 
                    fontsize=11, color='#34495E', bbox=dict(facecolor='white', edgecolor='none', alpha=0.85, pad=2), zorder=4)

    for nid, node in nodes.items():
        x, y = node['x'], node['y']
        shape = node['shape']
        color = node['color']
        label_text = '\n'.join(node['label_lines'])
        w, h = node['w'], node['h']
        
        font_size = 12
        if len(label_text) > 30: font_size = 11
        if len(label_text) > 45: font_size = 10
        if len(label_text) > 60: font_size = 9

        patch = None
        if shape == 'B':
            patch = patches.FancyBboxPatch((x-w/2, y-h/2), w, h, boxstyle="square,pad=0.05", facecolor=color, edgecolor='#2C3E50', lw=2, zorder=2)
        elif shape == 'C':
            r = node['r']
            patch = patches.Circle((x, y), radius=r, facecolor=color, edgecolor='#2C3E50', lw=2, zorder=2)
        elif shape == 'E':
            patch = patches.Ellipse((x, y), width=w, height=h, facecolor=color, edgecolor='#2C3E50', lw=2, zorder=2)
        elif shape == 'D':
            pad = 25
            rx, ry = w/2 + pad, h/2 + pad
            verts = [(x, y-ry), (x+rx, y), (x, y+ry), (x-rx, y)]
            patch = patches.Polygon(verts, closed=True, facecolor=color, edgecolor='#2C3E50', lw=2, zorder=2)
        elif shape == 'I':
            offset = w / 6
            verts = [(x-w/2+offset, y-h/2), (x+w/2+offset, y-h/2), (x+w/2-offset, y+h/2), (x-w/2-offset, y+h/2)]
            patch = patches.Polygon(verts, closed=True, facecolor=color, edgecolor='#2C3E50', lw=2, zorder=2)
            
        if patch:
            ax.add_patch(patch)
        ax.text(x, y, label_text, ha='center', va='center', fontsize=font_size, fontweight='bold', color='#111', zorder=3)

    buf = BytesIO()
    plt.savefig(buf, format='png', dpi=150, bbox_inches='tight', transparent=True, facecolor='none')
    plt.close(fig)
    return Image.open(buf)

def on_image_change(image):
    if image is None: return None
    return resize_image(image.convert("RGB"))

def on_extract_dsl(image, current_dsl):
    if image is None:
        return current_dsl, current_dsl, None
    MODEL.set_adapter("dsl_adapter")
    dsl_text = run_inference(UGDSL_PROMPT, image, GENERATION_KWARGS["dsl"], is_dsl=True)
    preview = render_dsl(dsl_text)
    return dsl_text, dsl_text, preview

def on_chat(message, history, mode, image, current_dsl):
    if not message.strip(): 
        return history or [], current_dsl, current_dsl, None
    
    history = history or []
    history.append({"role": "user", "content": message})

    if mode == "Вопрос-ответ":
        if image is None:
            history.append({"role": "assistant", "content": "Для режима Вопрос-ответ требуется изображение."})
            return history, current_dsl, current_dsl, None
        MODEL.set_adapter("qa_adapter")
        prompt = f"{message}\n{QUESTION_SUFFIX}"
        answer = run_inference(prompt, image, GENERATION_KWARGS["qa"])
        history.append({"role": "assistant", "content": answer})
        return history, current_dsl, current_dsl, None

    else:
        if not current_dsl or not current_dsl.strip().startswith("N "):
            history.append({"role": "assistant", "content": "Для режима редактирования сначала нужен извлечённый DSL-код."})
            return history, current_dsl, current_dsl, None
        MODEL.set_adapter("edit_adapter")
        edit_prompt = f"{message}\n{EDIT_UGDSL_SUFFIX}"
        full_prompt = f"{current_dsl}\n\n{edit_prompt}"
        new_dsl = run_inference(full_prompt, None, GENERATION_KWARGS["edit"], is_dsl=True)
        preview = render_dsl(new_dsl)
        history.append({"role": "assistant", "content": "DSL-код обновлён"})
        return history, new_dsl, new_dsl, preview

def on_dsl_manual_edit(new_text, current_dsl_state):
    preview = render_dsl(new_text)
    return new_text, new_text, preview

def clear_chat():
    return [], "", "", None

with gr.Blocks(title="(2B) VLM DSL и помощник по вопросам") as demo:
    gr.Markdown("Интерфейс модели VLM с адаптерами (2B bf16)")
    
    img_state = gr.State(None)
    dsl_state = gr.State("")

    with gr.Row():
        with gr.Column(scale=1, min_width=400):
            img_display = gr.Image(type="pil", sources=["upload", "clipboard"], label="Входное изображение", height=320)
            btn_extract = gr.Button("Извлечь DSL-код", variant="primary")
            dsl_box = gr.Textbox(label="Текущий DSL-код", lines=12, interactive=True, placeholder="Здесь появится DSL-вывод...", value="")
            gr.Markdown("*Ручные правки сохраняются для следующего запроса на редактирование.*")
            
        with gr.Column(scale=2):
            mode_radio = gr.Radio(["Вопрос-ответ", "Редактирование"], value="Вопрос-ответ", label="Режим работы")
            chatbot = gr.Chatbot(type="messages", height=300, allow_tags=False)
            with gr.Row():
                msg_input = gr.Textbox(placeholder="Введите вопрос или команду редактирования...", lines=2, container=False, scale=4)
                btn_send = gr.Button("Отправить", variant="primary", scale=1)
            
            dsl_preview = gr.Image(label="Визуализация DSL", interactive=False, type="pil", height=350)
            
            btn_clear = gr.Button("Очистить чат", variant="secondary")

    img_display.change(fn=on_image_change, inputs=[img_display], outputs=[img_state])
    btn_extract.click(fn=on_extract_dsl, inputs=[img_state, dsl_state], outputs=[dsl_state, dsl_box, dsl_preview])
    
    chat_inputs = [msg_input, chatbot, mode_radio, img_state, dsl_state]
    chat_outputs = [chatbot, dsl_state, dsl_box, dsl_preview]
    
    btn_send.click(fn=on_chat, inputs=chat_inputs, outputs=chat_outputs)
    msg_input.submit(fn=on_chat, inputs=chat_inputs, outputs=chat_outputs)
    
    dsl_box.change(fn=on_dsl_manual_edit, inputs=[dsl_box, dsl_state], outputs=[dsl_state, dsl_box, dsl_preview])
    btn_clear.click(fn=clear_chat, outputs=[chatbot, dsl_state, dsl_box, dsl_preview])

if __name__ == "__main__":
    init_model()
    demo.queue(default_concurrency_limit=1).launch(server_name="127.0.0.1", server_port=7860, share=False, inbrowser=True)