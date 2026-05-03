from huggingface_hub import snapshot_download, login
from pathlib import Path
import os

os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

PROJECT_ROOT = Path(__file__).parent
MODELS_DIR = PROJECT_ROOT / "models"

MODELS = {
#    "teacher": {
#        "name": "Qwen/Qwen3-VL-8B-Instruct",
#        "repo_id": "Qwen/Qwen3-VL-8B-Instruct",
#        "subfolder": "teacher",
#        "description": "Techer",
 #   },
  #  "analyzer-2B": {
   #     "name": "Qwen/Qwen3-VL-2B-Instruct",
    #    "repo_id": "Qwen/Qwen3-VL-2B-Instruct",
     #   "subfolder": "analyzer-2B",
      #  "description": "Analyzer",
    #},
        "analyzer-4B": {
        "name": "Qwen/Qwen3-VL-4B-Instruct",
        "repo_id": "Qwen/Qwen3-VL-4B-Instruct",
        "subfolder": "analyzer-4B",
        "description": "Analyzer",
    }
}
    
try:
    login(token=HF_TOKEN, add_to_git_credential=False)
    print("Logged in to HuggingFace")
except Exception as e:
    print(f"Login warning: {e}")

for model_key in MODELS:
    print(f"##### DOWNLOADING {model_key} #####")
    model = MODELS[model_key]
    model_path = MODELS_DIR / model['subfolder']
    snapshot_download(
        repo_id=model['repo_id'],
        local_dir=str(model_path),
        max_workers=8,
        token=HF_TOKEN,
    )