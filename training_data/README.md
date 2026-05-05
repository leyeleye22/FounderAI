# Qwen3 Fine-Tuning for Teranga Power Copilot

## Overview

This directory contains everything needed to fine-tune the Qwen3 model for the Teranga Power FounderAI copilot. The fine-tuning process teaches the model domain-specific knowledge about startup validation, business modeling, and the Teranga Power platform's 15 modules.

## Directory Structure

```
training_data/
├── generate_dataset.py          # Script to generate the JSONL training dataset
├── teranga_finetune.jsonl       # Generated training data (18 examples)
├── train_qwen3_lora.py          # LoRA fine-tuning script
├── merge_lora.py                # Script to merge LoRA adapter with base model
├── training_requirements.txt    # Python dependencies for training
└── README.md                    # This file
```

## Dataset

The training dataset (`teranga_finetune.jsonl`) covers all Teranga Power modules:

| Module | Examples | Focus |
|--------|----------|-------|
| problem-statement | 2 | Problem formulation, specificity, WHO/WHEN/COST |
| problem-validation | 2 | Urgency signals, validation criteria |
| icp | 1 | Ideal Customer Profile refinement |
| market-sizing | 1 | TAM/SAM/SOM calculation methods |
| business-model-canvas | 2 | BMC coherence, 9-block completion |
| go-to-market | 1 | GTM strategy phases, channels |
| competitive-landscape | 2 | Competitor analysis, positioning |
| roi | 2 | ROI calculation, business metrics (MRR, LTV, CAC) |
| user-journey | 2 | Journey mapping, friction resolution |
| research | 2 | Interview analysis, signal extraction |
| sprints | 1 | Sprint planning, task prioritization |
| gamma | 1 | Pitch deck structure |
| workshop | 1 | Workshop facilitation |

Each example is in conversation format with:
- `system`: Role definition and module context
- `user`: User query with module context and field data
- `assistant`: Expert response with concrete actions

## Training Pipeline

### 1. Install Dependencies

```bash
pip install -r training_requirements.txt
```

### 2. Generate Dataset (Optional)

```bash
python generate_dataset.py
```

### 3. Run Training

```bash
python train_qwen3_lora.py
```

The training script will:
1. Load the Qwen3 FP32 training base model from `./base_model_fp32`
2. Load the training dataset
3. Apply LoRA configuration (r=16, alpha=32)
4. Train for 3 epochs with batch size 1, gradient accumulation 8
5. Save the LoRA adapter to `./lora_adapter`

Note:
- `./` remains the main app/inference workspace.
- `./base_model_fp32` is the local FP32 training copy derived from the quantized base model.

### 3b. Relay Training for Weak Machines

If the PC cannot survive a full training run, use the relay workflow:

```bash
python build_relay_curriculum.py
python relay_train_qwen3_lora.py
```

How it works:
- `build_relay_curriculum.py` analyzes the merged dataset and splits the train set into progressive shards.
- `relay_train_qwen3_lora.py` trains only one tiny shard per run, saves the LoRA adapter, updates `lora_adapter_relay/relay_state.json`, and exits.
- You can run it again later and it resumes from the next shard like a relay race instead of restarting from zero.

Relay stages:
- `warmup`: short and structured examples, low `max_seq_length`, cheap CPU sessions.
- `build-up`: medium analytical tasks for stronger reasoning.
- `integration`: longest and most complex examples, including native Teranga patterns.

Relay artifacts:
- `training_data/relay_dataset_analysis.json`
- `training_data/relay_curriculum.json`
- `lora_adapter_relay/relay_state.json`
- `lora_adapter_relay/relay_history.json`

Recommended usage on CPU:
- Run one relay session when the machine is idle.
- Keep other heavy apps closed.
- Let sessions accumulate over days instead of forcing a single impossible run.

Useful micro-session overrides for very weak machines:

```powershell
$env:FOUNDER_AI_RELAY_MAX_STEPS='1'
$env:FOUNDER_AI_RELAY_MAX_SEQ_LENGTH='256'
$env:FOUNDER_AI_RELAY_QUICK_EVAL_SIZE='0'
.\.venv\Scripts\python.exe training_data\relay_train_qwen3_lora.py
```

The relay trainer now reopens an existing LoRA adapter in trainable mode, so resume sessions continue real learning instead of loading a frozen adapter.

### 3c. Overnight Relay Runner

To let the machine chain multiple relay sessions overnight, use:

```bash
python overnight_relay_runner.py --max-sessions 3
```

Useful options:
- `--start-hour 22 --end-hour 6`: only run inside the overnight window.
- `--min-idle-seconds 900`: require 15 minutes without keyboard or mouse input.
- `--max-prelaunch-cpu-percent 35`: avoid starting if another workload is already active.
- `--allow-battery`: disable the default AC-power safety check.
- `--dry-run`: preview availability checks and the next relay launch without committing to a real training session.

The overnight runner writes logs to:
- `lora_adapter_relay/overnight_runner.log`

### 3d. Behavior Repair Dataset

When you spot weak answers in real simulations, add corrected examples to:

- `training_data/behavior_repair_dataset.jsonl`

This file is merged automatically by `merge_datasets.py` when present. It is intended for:
- bad beginner prompts
- weak or unsafe model answers
- stronger expected answers used as repair targets

Current repairs cover:
- vague problem statements in sensitive health contexts
- validation based on family bias
- continent-scale ICPs
- customer segment mismatch
- top-down market sizing
- feature-first GTM prompts

### 3e. Google Colab Free V1

If your local PC does not have enough GPU or RAM, use the Colab notebook:

- `training_data/founderai_colab_free_v1.ipynb`
- `training_data/founderai_colab_train_eval_v2.ipynb`

What this setup changes compared with the local script:

- the notebook is standalone: dataset + helper scripts are embedded directly in the notebook
- uses a Hugging Face model id instead of a local Windows path
- uses 4-bit QLoRA by default
- uses a shorter sequence length (`512`) to fit smaller Colab GPUs
- writes checkpoints and metrics to local Colab storage by default
- resumes automatically from the latest checkpoint if the session was interrupted
- packages the adapter as a zip at the end of the notebook run

Recommended first pass on Colab free:

- runtime: `T4 GPU` if available
- `FOUNDER_AI_COLAB_USE_4BIT=true`
- `FOUNDER_AI_COLAB_EPOCHS=1`
- `FOUNDER_AI_COLAB_MAX_SEQ_LENGTH=512`
- `FOUNDER_AI_COLAB_GRAD_ACCUM=8`

If you only want a very fast smoke test before a fuller run:

- set `FOUNDER_AI_COLAB_SAMPLE_LIMIT=120`

If the session survives and memory is stable:

- set `FOUNDER_AI_COLAB_SAMPLE_LIMIT=0` to use the full train split
- optionally raise `FOUNDER_AI_COLAB_MAX_SEQ_LENGTH` to `768`

Entry point used by the notebook:

```bash
python training_data/train_qwen3_lora_colab.py
```

Outputs produced by the upgraded training flow:

- `training_metrics.json`
- `training_history.json`
- `training_report.md`
- `loss_curve.png`

The evaluation now includes:

- train loss
- validation loss
- test loss
- validation/test perplexity
- automatic overfit risk analysis
- a saved loss curve image

Recommended full retrain rerun for the current merged dataset:

- `FOUNDER_AI_COLAB_SAMPLE_LIMIT=0` to use the full train split
- `FOUNDER_AI_COLAB_EPOCHS=2` for a stronger second pass
- `FOUNDER_AI_COLAB_EVAL_STEPS=10`
- `FOUNDER_AI_COLAB_SAVE_STEPS=10`

This gives more intermediate evaluation points, which makes the overfit analysis more useful than the first smoke-test run.

Main environment variables for the Colab script:

- `FOUNDER_AI_COLAB_BASE_MODEL`
- `FOUNDER_AI_COLAB_DATA_PATH`
- `FOUNDER_AI_COLAB_OUTPUT_DIR`
- `FOUNDER_AI_COLAB_METRICS_PATH`
- `FOUNDER_AI_COLAB_SAMPLE_LIMIT`
- `FOUNDER_AI_COLAB_USE_4BIT`
- `FOUNDER_AI_COLAB_EPOCHS`
- `FOUNDER_AI_COLAB_MAX_SEQ_LENGTH`
- `FOUNDER_AI_COLAB_GRAD_ACCUM`
- `FOUNDER_AI_COLAB_SAVE_STEPS`
- `FOUNDER_AI_COLAB_EVAL_STEPS`

Notes for Colab free:

- runtime availability is dynamic
- GPU access is not guaranteed
- sessions can terminate early
- the notebook is optimized for a quick first run without extra setup

### 3f. Publish a Colab adapter to Hugging Face Hub

Once Colab generates a zip like `founderai_lora_adapter.zip`, you can publish it to a Hugging Face model repo:

```powershell
cd "C:\Users\Mr LEYE\Downloads\FounderAI"
$env:HF_TOKEN="hf_xxx"
python scripts/publish_lora_adapter_to_hub.py `
  "C:\Users\Mr LEYE\Downloads\founderai_lora_adapter.zip" `
  --repo-id "leyeleye22/founderai-qwen3-lora-v1"
```

This is useful when:

- you train only from time to time on Colab
- you want a clean registry of adapters
- you want GitHub / deployment infrastructure to reference a stable Hub repo instead of random local zip files

### 4. Merge Adapter (Optional)

```bash
python merge_lora.py
```

This creates a standalone merged model at `./teranga-qwen3-merged`.

## Integration with FounderAI Backend

After training, enable the fine-tuned model by updating the `.env` file:

```env
USE_FINETUNED_MODEL=true
FINETUNED_MODEL_PATH=.
LORA_ADAPTER_PATH=./lora_adapter
```

The backend will automatically:
1. Try to load the fine-tuned model with LoRA adapter
2. Fall back to API-based Qwen if model loading fails
3. Fall back to heuristics if API is unavailable

## Configuration

### Training Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `lora_r` | 16 | LoRA attention dimension |
| `lora_alpha` | 32 | LoRA scaling factor |
| `lora_dropout` | 0.05 | LoRA dropout rate |
| `num_train_epochs` | 3 | Number of training epochs |
| `learning_rate` | 2e-4 | Learning rate |
| `max_seq_length` | 2048 | Maximum sequence length |
| `gradient_accumulation_steps` | 8 | Gradient accumulation steps |

### Target Modules

LoRA is applied to the following model layers:
- `q_proj`, `k_proj`, `v_proj`, `o_proj` (attention)
- `gate_proj`, `up_proj`, `down_proj` (FFN)

## Hardware Requirements

- **Minimum**: 8GB VRAM (4-bit quantization)
- **Recommended**: 16GB+ VRAM (8-bit or full precision)
- **CPU Training**: Possible but extremely slow (not recommended)

## Expanding the Dataset

To add more training examples:

1. Edit `generate_dataset.py` and add new entries to the `EXAMPLES` list
2. Follow the conversation format:
   ```python
   {
       "messages": [
           {"role": "system", "content": "..."},
           {"role": "user", "content": "..."},
           {"role": "assistant", "content": "..."},
       ]
   }
   ```
3. Regenerate the dataset: `python generate_dataset.py`

## Best Practices

1. **Start small**: Begin with 18 examples, validate quality, then expand
2. **Balance modules**: Ensure each module has adequate representation
3. **Realistic inputs**: Use actual field data from the Teranga Power frontend
4. **Action-oriented outputs**: Responses should always propose concrete next steps
5. **No hallucination**: Never invent numbers - always cite sources or methods
