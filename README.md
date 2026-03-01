# LLM Stack Monorepo (8GB GPU Optimized)

```text
llm-stack/
├── services/ (training, serving, data)
├── evaluation/
├── observability/
├── infra/
├── tests/
├── configs/
└── .github/workflows/
```

## Architecture
- **Training service**: pretraining, SFT, LoRA, DPO with modular entry points.
- **Serving service**: FastAPI interface with optional vLLM backend.
- **Data service**: streaming ingestion + tokenizer + caching.
- **Observability**: JSON logs, Prometheus gauges, optional MLflow logging.
- **Infra**: Dockerfiles, Compose, K8s manifests, cloud notes.

## 8GB Memory Strategy
- ~300M decoder-only config (12L, 1024 hidden, 8 heads, seq 512)
- FP16/BF16 mixed precision
- gradient accumulation (default 16)
- micro-batch size 1
- optional gradient checkpointing and LoRA adapters

Estimated model params are exposed via `estimate_params()`.

## Quickstart
```bash
make install
make preprocess
make pretrain
make sft
make lora
make dpo
make evaluate
make merge
make serve
```

## Training from Scratch
`services/training/pretrain/train.py` runs next-token training with AdamW, cosine decay, checkpointing.

## Fine-Tuning
- **SFT**: response-only masked loss via `build_sft_labels`.
- **LoRA**: `apply_lora`, `save_adapter`, and `merge_adapter`.
- **DPO**: `dpo_loss` with chosen/rejected preference pairs.

## Serving
Start locally:
```bash
docker compose -f infra/compose/docker-compose.yml up serving
```
Endpoints: `/health`, `/version`, `/generate`.

## Cloud deployment
- Apply `infra/k8s/serving-deployment.yaml`.
- Configure object storage via `S3_ENDPOINT`/`S3_BUCKET`.

## CI policy
PRs to `main` run lint, mypy, pytest, Docker builds, minimal training check, and inference endpoint validation. Any failure blocks merge.


## Testing and validation
- Local: `make lint` then `make test`.
- CI: `.github/workflows/ci.yml` runs lint, mypy, pytest, Docker image builds, a minimal pretrain smoke run, and serving health validation.
- If your environment lacks ML/runtime dependencies (e.g., `torch`/`fastapi`), run inside Docker or install extras with `make install`.
