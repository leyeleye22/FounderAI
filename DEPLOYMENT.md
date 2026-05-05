# FounderAI Deployment and CI/CD Plan

## Goal

Deploy FounderAI as a stable API service while keeping trained model updates releasable without rebuilding the whole application around local machine paths.

## Recommended Architecture

Use three distinct layers:

1. Front and proxy on Vercel
2. Laravel API on Hostinger
3. FounderAI as a separate Python service

Runtime flow:

`Browser -> Next proxy on Vercel -> FounderAI public URL -> Laravel API + LLM backend`

## Why separate the AI service

FounderAI is not just a thin endpoint:

- it is a FastAPI application
- it contains retrieval logic
- it can point to local or remote model assets
- it may evolve independently from the Next frontend

That makes a container-based deployment model a better fit than a serverless-only setup.

## Deployment Model

### Application release

The FounderAI API is deployed as a Docker image:

- built from `Dockerfile`
- pushed to `ghcr.io/<owner>/founderai`
- deployed by your runtime platform from that image

### Model release

The trained model update is released separately as a versioned bundle:

- produced by `scripts/build_model_bundle.py`
- uploaded by `.github/workflows/model-bundle.yml`
- includes training data metadata, adapter folders when present, and config files

This separation matters because:

- app code changes should not require rebuilding giant model artifacts
- model updates should not require changing application code
- rollback becomes easier because app image version and model bundle version can move independently

## Environments

Use at least:

- `local`
- `staging`
- `production`

## Required Environment Variables

### FounderAI service

- `FOUNDER_AI_ENV`
- `FOUNDER_AI_HOST`
- `FOUNDER_AI_PORT`
- `FOUNDER_AI_MODEL_DIR`
- `FOUNDER_AI_RAG_DIR`
- `FOUNDER_AI_FORCE_IN_MEMORY_RETRIEVAL`
- `FOUNDERPATH_API_BASE_URL`
- `FOUNDERPATH_API_TOKEN`
- `FOUNDER_AI_USE_LOCAL_HEURISTICS`
- `LLM_PROVIDER`
- `LLM_API_BASE_URL`
- `LLM_MODEL_NAME`
- `HF_INFERENCE_MODEL`
- `HF_INFERENCE_PROVIDER`
- `HF_TOKEN`
- `USE_FINETUNED_MODEL`
- `FINETUNED_MODEL_PATH`
- `LORA_ADAPTER_PATH`

### Frontend on Vercel

- `FOUNDER_AI_BASE_URL`
- `APP_API_BASE_URL`

Important:

- production should not rely on `127.0.0.1`
- production should not rely on local Windows file paths

## CI Pipeline

Workflow: `.github/workflows/ci.yml`

Checks:

- install Python dependencies
- compile Python files
- run `pytest`
- validate API contract through route and chat tests

This protects the frontend contract for:

- `/health`
- `/agents/catalog`
- `/agents/problem/challenge`
- `/agents/interview/analyze`
- `/agents/sprint/plan`
- `/agents/chat`

## Container Release Pipeline

Workflow: `.github/workflows/container-release.yml`

Trigger:

- push to `main`
- version tags like `v1.0.0`
- manual dispatch

Output:

- Docker image pushed to GHCR
- immutable image tags based on commit SHA
- `latest` tag for the default branch

## Model Bundle Pipeline

Workflow: `.github/workflows/model-bundle.yml`

Trigger:

- changes in training or adapter folders on `main`
- manual dispatch with optional bundle version

Output:

- versioned manifest
- `.tar.gz` bundle in `dist/model-bundles`
- uploaded workflow artifact

## Recommended Rollout Strategy

### API rollout

1. Merge to `main`
2. CI must pass
3. Container image is built and pushed
4. Deploy staging from the new image
5. Run smoke tests
6. Promote to production

### Model rollout

1. Train or update adapter
2. Commit training metadata and packaging logic as needed
3. Run `model-bundle` workflow
4. Publish or store the produced bundle
5. Mount or copy bundle to the deployment target
6. Update:
   - `USE_FINETUNED_MODEL=true`
   - `FINETUNED_MODEL_PATH`
   - `LORA_ADAPTER_PATH`
7. Restart FounderAI
8. Run smoke tests against `/agents/chat`

## Recommended Smoke Tests

At minimum after deployment:

- `GET /health`
- `GET /agents/catalog`
- `POST /agents/chat` with a valid payload from the frontend contract

## Rollback Strategy

Keep rollback simple:

- revert to previous Docker image tag for application issues
- revert to previous model bundle version for model-quality regressions

This is exactly why app image and model bundle should be versioned independently.

## Practical V1 Recommendation

For V1, use this stack:

- Front proxy: Vercel
- Laravel: Hostinger
- FounderAI API: container host or VPS
- Model runtime: remote OpenAI-compatible endpoint first, local fine-tuned bundle later

This gives you a clean path:

- ship API now
- train and improve the model later
- deploy model updates without redesigning the system

## Vercel V1 Mode

When you need a fast V1 on Vercel before moving to a VPS:

- deploy this repository to Vercel as a Python project
- use `api/index.py` as the runtime entrypoint
- set `FOUNDER_AI_BASE_URL` in the Next proxy to `https://<your-founder-ai>.vercel.app/api`
- set `FOUNDER_AI_FORCE_IN_MEMORY_RETRIEVAL=true`
- keep `USE_FINETUNED_MODEL=false`
- keep `FOUNDER_AI_USE_LOCAL_HEURISTICS=true` unless you connect a remote `LLM_API_BASE_URL`

Important for Vercel:

- `requirements.txt` is intentionally lightweight for the serverless runtime
- retrieval extras live in `requirements.retrieval.txt`
- local model serving should wait for your future Hostinger VPS target

## Hugging Face V1

If you want a very fast V1 before serving your own fine-tuned model, FounderAI can use Hugging Face hosted inference directly.

Use these `.env` values:

```env
LLM_PROVIDER=huggingface
HF_INFERENCE_MODEL=Qwen/Qwen3-8B
HF_INFERENCE_PROVIDER=auto
HF_TOKEN=
USE_FINETUNED_MODEL=false
```

What this gives you:

- FounderAI keeps its own business logic and response shaping
- only the text generation layer is delegated to Hugging Face
- your frontend contract stays unchanged

What this does not give you:

- it does not automatically serve your Colab LoRA adapter
- it is best for a fast remote baseline while your fine-tuned flow matures

## Colab Adapter -> Hugging Face Hub

When a Colab run finishes and gives you a zip like `founderai_lora_adapter.zip`, publish it to a Hugging Face model repo:

```powershell
cd "C:\Users\Mr LEYE\Downloads\FounderAI"
$env:HF_TOKEN="hf_xxx"
python scripts/publish_lora_adapter_to_hub.py `
  "C:\Users\Mr LEYE\Downloads\founderai_lora_adapter.zip" `
  --repo-id "leyeleye22/founderai-qwen3-lora-v1"
```

This script:

- accepts either a local adapter folder or the Colab zip directly
- validates required adapter files
- creates the Hub model repo if needed
- uploads the adapter and a basic model card

## GitHub Actions for Hugging Face publishing

A manual workflow is included:

- `.github/workflows/huggingface-model-publish.yml`

Use it when the adapter folder or zip is already available in the checked-out workspace on GitHub Actions.

Required secret:

- `HF_TOKEN`

Useful note:

- because Colab artifacts are usually downloaded to your laptop first, the local script is the main path after training
- the GitHub Actions workflow is most useful when you intentionally place the adapter in the repo or in a pre-staged workspace

## Local LoRA Smoke Test

When you download a Colab adapter zip such as `founderai_lora_adapter.zip`, install it locally before testing FounderAI:

```powershell
cd "C:\Users\Mr LEYE\Downloads\FounderAI"
python scripts/install_lora_adapter.py "C:\Users\Mr LEYE\Downloads\founderai_lora_adapter.zip" --force
```

This will extract the root adapter files into:

- `models/founderai/current/lora_adapter`

Then configure your local `.env` with:

```env
USE_FINETUNED_MODEL=true
FINETUNED_MODEL_PATH=Qwen/Qwen3-4B
LORA_ADAPTER_PATH=C:\Users\Mr LEYE\Downloads\FounderAI\models\founderai\current\lora_adapter
HF_TOKEN=
```

Notes:

- `FINETUNED_MODEL_PATH` can now be either a local model folder or a Hugging Face model id such as `Qwen/Qwen3-4B`
- set `HF_TOKEN` if you hit anonymous rate limits on Hugging Face
- keep `FOUNDER_AI_FORCE_IN_MEMORY_RETRIEVAL=true` for a lightweight local V1 if you do not want Chroma locally

Launch the API locally:

```powershell
cd "C:\Users\Mr LEYE\Downloads\FounderAI"
uvicorn app.main:app --host 0.0.0.0 --port 8010
```

Quick checks:

```powershell
Invoke-WebRequest http://127.0.0.1:8010/health
```

```powershell
$body = @{
  project = @{
    workspace_id = "demo-workspace"
    project_id = "demo-project"
  }
  module = @{
    module_key = "problem_statement"
    label = "Problem Statement"
    filled_fields = @(
      @{
        field_name = "problemStatement"
        label = "Problem Statement"
        is_filled = $true
        content = "Freelance founders lose time rewriting weak problem statements."
      }
    )
    empty_fields = @("who", "when", "cost")
    raw_content = "Freelance founders lose time rewriting weak problem statements."
  }
  message = "Help me improve this problem statement and suggest stronger field values."
  locale = "fr"
  conversation_history = @()
} | ConvertTo-Json -Depth 10

Invoke-RestMethod `
  -Uri "http://127.0.0.1:8010/agents/chat" `
  -Method Post `
  -ContentType "application/json" `
  -Body $body
```

Expected result:

- a JSON response with `reply`
- `actions`
- `module_key`
- optionally `field_proposals`
