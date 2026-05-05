from __future__ import annotations

import base64
import gzip
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
NOTEBOOK_PATH = ROOT / "founderai_colab_free_v1.ipynb"
DATASET_PATH = ROOT / "teranga_merged.jsonl"
FINETUNE_UTILS_PATH = ROOT / "finetune_utils.py"
TRAIN_SCRIPT_PATH = ROOT / "train_qwen3_lora_colab.py"
TRAINING_REQUIREMENTS_PATH = ROOT / "training_requirements.txt"


def build_notebook() -> dict:
    dataset_bytes = DATASET_PATH.read_bytes()
    dataset_b64 = base64.b64encode(gzip.compress(dataset_bytes, compresslevel=9)).decode("ascii")
    finetune_utils = FINETUNE_UTILS_PATH.read_text(encoding="utf-8")
    train_script = TRAIN_SCRIPT_PATH.read_text(encoding="utf-8")
    requirements = [
        line.strip()
        for line in TRAINING_REQUIREMENTS_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]
    pip_line = "!pip install -q " + " ".join(requirements)

    cells = [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# FounderAI Colab Free V1\n",
                "\n",
                "Notebook standalone: dataset + scripts de training inclus. Une fois le notebook ouvert dans Colab, tu peux faire `Run all`.\n",
                "\n",
                "Avant de lancer:\n",
                "- Ouvre `Runtime > Change runtime type`\n",
                "- Choisis `T4 GPU` si disponible\n",
                "- Ensuite clique `Run all`\n",
            ],
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "import os\n",
                "from pathlib import Path\n",
                "\n",
                "RUN_ROOT = Path('/content/founderai-colab-v1')\n",
                "TRAINING_DIR = RUN_ROOT / 'training_data'\n",
                "OUTPUT_DIR = RUN_ROOT / 'lora_adapter'\n",
                "TRAINING_DIR.mkdir(parents=True, exist_ok=True)\n",
                "OUTPUT_DIR.mkdir(parents=True, exist_ok=True)\n",
                "print('Run root:', RUN_ROOT)\n",
            ],
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [pip_line + "\n"],
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": ["!nvidia-smi\n"],
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "import base64\n",
                "import gzip\n",
                "from pathlib import Path\n",
                "\n",
                f"DATASET_B64 = '''{dataset_b64}'''\n",
                "dataset_path = Path('/content/founderai-colab-v1/training_data/teranga_merged.jsonl')\n",
                "dataset_path.write_bytes(gzip.decompress(base64.b64decode(DATASET_B64)))\n",
                "print('Dataset written to:', dataset_path)\n",
                "print('Dataset size bytes:', dataset_path.stat().st_size)\n",
            ],
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "from pathlib import Path\n",
                "\n",
                "finetune_utils_path = Path('/content/founderai-colab-v1/training_data/finetune_utils.py')\n",
                "train_script_path = Path('/content/founderai-colab-v1/training_data/train_qwen3_lora_colab.py')\n",
                f"finetune_utils_path.write_text({finetune_utils!r}, encoding='utf-8')\n",
                f"train_script_path.write_text({train_script!r}, encoding='utf-8')\n",
                "print('Wrote helper scripts.')\n",
            ],
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "import os\n",
                "\n",
                "os.environ['FOUNDER_AI_COLAB_BASE_MODEL'] = 'Qwen/Qwen3-4B'\n",
                "os.environ['FOUNDER_AI_COLAB_DATA_PATH'] = '/content/founderai-colab-v1/training_data/teranga_merged.jsonl'\n",
                "os.environ['FOUNDER_AI_COLAB_OUTPUT_DIR'] = '/content/founderai-colab-v1/lora_adapter'\n",
                "os.environ['FOUNDER_AI_COLAB_METRICS_PATH'] = '/content/founderai-colab-v1/lora_adapter/training_metrics.json'\n",
                "os.environ['FOUNDER_AI_COLAB_USE_4BIT'] = 'true'\n",
                "os.environ['FOUNDER_AI_COLAB_EPOCHS'] = '1'\n",
                "os.environ['FOUNDER_AI_COLAB_MAX_SEQ_LENGTH'] = '512'\n",
                "os.environ['FOUNDER_AI_COLAB_GRAD_ACCUM'] = '8'\n",
                "os.environ['FOUNDER_AI_COLAB_SAVE_STEPS'] = '25'\n",
                "os.environ['FOUNDER_AI_COLAB_EVAL_STEPS'] = '25'\n",
                "os.environ['FOUNDER_AI_COLAB_SAMPLE_LIMIT'] = '120'\n",
                "print('Training config ready. First run will use 120 samples for safety on Colab free.')\n",
            ],
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "import subprocess\n",
                "from pathlib import Path\n",
                "\n",
                "log_path = Path('/content/founderai-colab-v1/training_run.log')\n",
                "with log_path.open('w', encoding='utf-8') as log_file:\n",
                "    result = subprocess.run(\n",
                "        ['python', '/content/founderai-colab-v1/training_data/train_qwen3_lora_colab.py'],\n",
                "        stdout=log_file,\n",
                "        stderr=subprocess.STDOUT,\n",
                "        text=True,\n",
                "        check=False,\n",
                "    )\n",
                "print(log_path.read_text(encoding='utf-8')[:20000])\n",
                "if result.returncode != 0:\n",
                "    raise RuntimeError(f'Training failed with exit code {result.returncode}. Full log saved at {log_path}')\n",
            ],
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "!ls -lah /content/founderai-colab-v1/lora_adapter\n",
                "!find /content/founderai-colab-v1/lora_adapter -maxdepth 2 -type f | sort\n",
            ],
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "import json\n",
                "from pathlib import Path\n",
                "\n",
                "metrics_path = Path('/content/founderai-colab-v1/lora_adapter/training_metrics.json')\n",
                "if not metrics_path.exists():\n",
                "    raise FileNotFoundError(f'Missing training metrics at {metrics_path}')\n",
                "metrics = json.loads(metrics_path.read_text(encoding='utf-8'))\n",
                "print(json.dumps(metrics, ensure_ascii=False, indent=2)[:12000])\n",
            ],
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "from pathlib import Path\n",
                "\n",
                "output_dir = Path('/content/founderai-colab-v1/lora_adapter')\n",
                "required = [output_dir / 'adapter_config.json', output_dir / 'tokenizer_config.json']\n",
                "weight_candidates = [output_dir / 'adapter_model.safetensors', output_dir / 'adapter_model.bin']\n",
                "missing = [str(path) for path in required if not path.exists()]\n",
                "if not any(path.exists() for path in weight_candidates):\n",
                "    missing.append('adapter_model.safetensors or adapter_model.bin')\n",
                "if missing:\n",
                "    found = sorted(path.name for path in output_dir.iterdir()) if output_dir.exists() else []\n",
                "    raise RuntimeError(f'Adapter output is incomplete. Missing: {missing}. Found: {found}')\n",
                "print('Adapter artifacts look valid.')\n",
            ],
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "import shutil\n",
                "from google.colab import files\n",
                "\n",
                "archive_path = shutil.make_archive('/content/founderai-colab-v1/founderai_lora_adapter', 'zip', '/content/founderai-colab-v1/lora_adapter')\n",
                "print('Created archive:', archive_path)\n",
                "files.download(archive_path)\n",
            ],
        },
    ]

    return {
        "cells": cells,
        "metadata": {
            "accelerator": "GPU",
            "colab": {"gpuType": "T4", "provenance": []},
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def main() -> None:
    notebook = build_notebook()
    NOTEBOOK_PATH.write_text(json.dumps(notebook, ensure_ascii=False), encoding="utf-8")
    print(f"Notebook regenerated: {NOTEBOOK_PATH}")
    print(f"Notebook size: {NOTEBOOK_PATH.stat().st_size} bytes")


if __name__ == "__main__":
    main()
