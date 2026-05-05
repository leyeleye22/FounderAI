from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
NOTEBOOK_PATH = ROOT / "founderai_colab_train_eval_v2.ipynb"


def build_notebook() -> dict:
    pip_line = (
        "!pip install -q "
        "transformers>=4.51.0 peft>=0.10.0 accelerate>=0.28.0 bitsandbytes>=0.43.0 "
        "datasets>=2.18.0 trl>=0.8.0 torch>=2.2.0 sentencepiece>=0.2.0 protobuf>=4.25.0 "
        "matplotlib>=3.8.0 fastapi==0.115.9 uvicorn[standard]==0.35.0 pydantic==2.11.7 "
        "pydantic-settings==2.10.1 httpx==0.28.1 huggingface_hub>=0.34.0"
    )

    cells = [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# FounderAI Colab Train + Eval V2\n",
                "\n",
                "Notebook dedie au retrain sur Colab avec evaluation complete:\n",
                "- train loss\n",
                "- validation loss\n",
                "- test loss\n",
                "- perplexity\n",
                "- signaux d'overfit\n",
                "- courbe de loss\n",
                "- eval comportementale problem statement\n",
                "- eval multi-modules (`validation`, `ICP`, `business`, `GTM`, `market sizing`, `ROI`, `research`, `journey`)\n",
                "\n",
                "Usage:\n",
                "1. Runtime > Change runtime type\n",
                "2. Choisis `T4 GPU` si disponible\n",
                "3. Clique `Run all`\n",
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
                "RUN_ROOT = Path('/content/founderai-colab-train-eval')\n",
                "REPO_URL = 'https://github.com/leyeleye22/FounderAI.git'\n",
                "\n",
                "if not RUN_ROOT.exists():\n",
                "    subprocess.run(['git', 'clone', REPO_URL, str(RUN_ROOT)], check=True)\n",
                "else:\n",
                "    print('Repo already present at', RUN_ROOT)\n",
                "\n",
                "OUTPUT_DIR = RUN_ROOT / 'colab_outputs' / 'lora_adapter'\n",
                "OUTPUT_DIR.mkdir(parents=True, exist_ok=True)\n",
                "print('Repo root:', RUN_ROOT)\n",
                "print('Output dir:', OUTPUT_DIR)\n",
            ],
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "import os\n",
                "os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'\n",
                "print('Set safer CUDA allocation mode for Colab GPUs.')\n",
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
                "import os\n",
                "\n",
                "os.environ['FOUNDER_AI_COLAB_BASE_MODEL'] = 'Qwen/Qwen3-4B'\n",
                "os.environ['FOUNDER_AI_COLAB_DATA_PATH'] = str(RUN_ROOT / 'training_data' / 'teranga_merged.jsonl')\n",
                "os.environ['FOUNDER_AI_COLAB_OUTPUT_DIR'] = str(OUTPUT_DIR)\n",
                "os.environ['FOUNDER_AI_COLAB_METRICS_PATH'] = str(OUTPUT_DIR / 'training_metrics.json')\n",
                "os.environ['FOUNDER_AI_COLAB_HISTORY_PATH'] = str(OUTPUT_DIR / 'training_history.json')\n",
                "os.environ['FOUNDER_AI_COLAB_REPORT_PATH'] = str(OUTPUT_DIR / 'training_report.md')\n",
                "os.environ['FOUNDER_AI_COLAB_PLOT_PATH'] = str(OUTPUT_DIR / 'loss_curve.png')\n",
                "os.environ['FOUNDER_AI_COLAB_EVAL_SUMMARY_JSON'] = str(OUTPUT_DIR / 'behavioral_eval_summary.json')\n",
                "os.environ['FOUNDER_AI_COLAB_EVAL_SUMMARY_MD'] = str(OUTPUT_DIR / 'behavioral_eval_summary.md')\n",
                "os.environ['FOUNDER_AI_COLAB_USE_4BIT'] = 'true'\n",
                "os.environ['FOUNDER_AI_COLAB_EPOCHS'] = '2'\n",
                "os.environ['FOUNDER_AI_COLAB_MAX_SEQ_LENGTH'] = '512'\n",
                "os.environ['FOUNDER_AI_COLAB_GRAD_ACCUM'] = '8'\n",
                "os.environ['FOUNDER_AI_COLAB_SAVE_STEPS'] = '10'\n",
                "os.environ['FOUNDER_AI_COLAB_EVAL_STEPS'] = '10'\n",
                "os.environ['FOUNDER_AI_COLAB_SAMPLE_LIMIT'] = '0'\n",
                "print('Training + eval config ready for the full train split.')\n",
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
                "log_path = OUTPUT_DIR / 'training_run.log'\n",
                "script_path = RUN_ROOT / 'training_data' / 'train_qwen3_lora_colab.py'\n",
                "\n",
                "with log_path.open('w', encoding='utf-8') as log_file:\n",
                "    result = subprocess.run(\n",
                "        ['python', str(script_path)],\n",
                "        cwd=str(RUN_ROOT),\n",
                "        stdout=log_file,\n",
                "        stderr=subprocess.STDOUT,\n",
                "        text=True,\n",
                "        check=False,\n",
                "    )\n",
                "\n",
                "log_text = log_path.read_text(encoding='utf-8')\n",
                "print(log_text[:12000])\n",
                "if result.returncode != 0:\n",
                "    print('\\n--- LOG TAIL ---\\n')\n",
                "    print(log_text[-12000:])\n",
                "    raise RuntimeError(f'Training failed with exit code {result.returncode}. Full log saved at {log_path}')\n",
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
                "metrics_path = OUTPUT_DIR / 'training_metrics.json'\n",
                "report_path = OUTPUT_DIR / 'training_report.md'\n",
                "history_path = OUTPUT_DIR / 'training_history.json'\n",
                "\n",
                "if not metrics_path.exists():\n",
                "    raise FileNotFoundError(f'Missing metrics file at {metrics_path}')\n",
                "\n",
                "metrics = json.loads(metrics_path.read_text(encoding='utf-8'))\n",
                "summary = {\n",
                "    'train_loss': metrics.get('train_loss'),\n",
                "    'validation_loss': metrics.get('validation_loss'),\n",
                "    'test_loss': metrics.get('test_loss'),\n",
                "    'validation_perplexity': metrics.get('validation_perplexity'),\n",
                "    'test_perplexity': metrics.get('test_perplexity'),\n",
                "    'overfit_analysis': metrics.get('overfit_analysis'),\n",
                "}\n",
                "print(json.dumps(summary, ensure_ascii=False, indent=2))\n",
            ],
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "from IPython.display import Image, Markdown, display\n",
                "\n",
                "plot_path = OUTPUT_DIR / 'loss_curve.png'\n",
                "report_path = OUTPUT_DIR / 'training_report.md'\n",
                "\n",
                "if plot_path.exists():\n",
                "    display(Image(filename=str(plot_path)))\n",
                "else:\n",
                "    print('No loss curve found.')\n",
                "\n",
                "if report_path.exists():\n",
                "    display(Markdown(report_path.read_text(encoding='utf-8')))\n",
                "else:\n",
                "    print('No markdown report found.')\n",
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
                "eval_log_path = OUTPUT_DIR / 'behavioral_eval.log'\n",
                "eval_script_path = RUN_ROOT / 'training_data' / 'run_colab_full_eval.py'\n",
                "\n",
                "with eval_log_path.open('w', encoding='utf-8') as log_file:\n",
                "    result = subprocess.run(\n",
                "        ['python', str(eval_script_path)],\n",
                "        cwd=str(RUN_ROOT),\n",
                "        stdout=log_file,\n",
                "        stderr=subprocess.STDOUT,\n",
                "        text=True,\n",
                "        check=False,\n",
                "    )\n",
                "\n",
                "log_text = eval_log_path.read_text(encoding='utf-8')\n",
                "print(log_text[:12000])\n",
                "if result.returncode != 0:\n",
                "    print('\\n--- EVAL LOG TAIL ---\\n')\n",
                "    print(log_text[-12000:])\n",
                "    raise RuntimeError(f'Behavioral eval failed with exit code {result.returncode}. Full log saved at {eval_log_path}')\n",
            ],
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "import json\n",
                "from IPython.display import Markdown, display\n",
                "\n",
                "behavior_json = OUTPUT_DIR / 'behavioral_eval_summary.json'\n",
                "behavior_md = OUTPUT_DIR / 'behavioral_eval_summary.md'\n",
                "\n",
                "if behavior_json.exists():\n",
                "    payload = json.loads(behavior_json.read_text(encoding='utf-8'))\n",
                "    compact = {\n",
                "        'problem_statement_eval': payload.get('problem_statement_eval'),\n",
                "        'conversational_eval': payload.get('conversational_eval'),\n",
                "        'top_failures': payload.get('top_failures', [])[:3],\n",
                "    }\n",
                "    print(json.dumps(compact, ensure_ascii=False, indent=2))\n",
                "else:\n",
                "    raise FileNotFoundError(f'Missing behavioral eval summary at {behavior_json}')\n",
                "\n",
                "if behavior_md.exists():\n",
                "    display(Markdown(behavior_md.read_text(encoding='utf-8')))\n",
                "else:\n",
                "    print('No behavioral eval markdown report found.')\n",
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
                "archive_path = shutil.make_archive(str(OUTPUT_DIR.parent / 'founderai_lora_adapter_train_eval_v2'), 'zip', str(OUTPUT_DIR))\n",
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
    NOTEBOOK_PATH.write_text("\ufeff" + json.dumps(notebook, ensure_ascii=False), encoding="utf-8")
    print(f"Notebook regenerated: {NOTEBOOK_PATH}")
    print(f"Notebook size: {NOTEBOOK_PATH.stat().st_size} bytes")


if __name__ == "__main__":
    main()
