# Troubleshooting

## Docker issues

**`/models: not found` during build**
The `models/` folder is excluded via `.dockerignore` and mounted as a volume
at runtime. Do not add `COPY models/` to the Dockerfile -- models are too
large to bake into images.

**Context transfer is very large (GBs)**
Check `.dockerignore` exists and includes `models/`, `wandb/`,
`unsloth_compiled_cache/`, and `datasets/raw/`. Without it, Docker copies
all model files into the build context.

**Storage fills up fast**
Run `docker system prune -af` to clear unused images and build cache.
Docker image layers can consume 20-50GB quickly during development.

## Backend issues

**`503 Model not loaded yet`**
Backend loads models at startup which takes 30-60s. Wait and retry.

**`FileNotFoundError` on startup**
Model checkpoints are not in the repo (gitignored). Ensure these exist locally:
- `models/sakshi_llama31_8b_rank4/checkpoint-9/`
- `models/qlora-primary-dpo/checkpoint-7/`
- `models/router.pkl`

**Out of memory (OOM) on model load**
Reduce `max_seq_length` in `HybridModelServer()` init, or ensure no other
GPU processes are running before starting the backend.

## Frontend issues

**`Could not connect to backend`**
Ensure the FastAPI backend is running on port 8000 before starting Streamlit.
If using Docker, both services must be up (`docker-compose up`).

**Generation times out (>120s)**
The teacher model loads lazily -- first teacher-routed request adds 30-60s
load time. Subsequent requests are faster.

## Training issues

**Windows Smart App Control blocks training scripts**
Known issue (Faiza, Week 7). Microsoft's only fix once Smart App Control is
fully enabled is a full Windows reset. Workaround: use WSL2 or another machine.

**Unsloth fused CE loss crashes on Windows**
Apply the monkey-patch in `training/train_qlora_sakshi.py` (already included).
Root cause: Windows WDDM returns near-zero free GPU memory, breaking the
auto-detection. The patch forces a safe fixed memory budget instead.
