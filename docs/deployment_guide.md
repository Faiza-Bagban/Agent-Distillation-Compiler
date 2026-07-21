# Deployment Guide -- Agent Distillation Compiler (Week 7 Fri)

## Prerequisites
- Python 3.10+ with conda env `adc` (see `docs/env_sakshi.md` for setup)
- NVIDIA GPU with CUDA 12.8 (RTX 4050 6GB minimum for student model alone)
- Docker Desktop (for sandbox executor)
- Ollama (for teacher fallback pipeline)

## Quick start (inference only)

1. Clone the repo and activate the environment:

git clone https://github.com/YeshitaMotwani/Agent-Distillation-Compiler.git
cd Agent-Distillation-Compiler
conda activate adc
pip install -r requirements.txt


2. Place model checkpoints (gitignored, not in repo):

models/
sakshi_llama31_8b_rank4/checkpoint-9/ # student model
qlora-primary-dpo/checkpoint-7/ # DPO-aligned teacher
router.pkl # complexity router


3. Start the FastAPI backend (Week 8):

uvicorn backend.main:app --host 0.0.0.0 --port 8000


4. Launch the Streamlit demo (Week 8):

streamlit run frontend/app.py


## One-command Docker deploy (Week 8)

docker-compose up

Starts backend + frontend together. See `docker-compose.yml` for port config.

## Model files
Models are NOT in the repo (too large). Team members share via:
- WhatsApp/Drive zip for DPO checkpoint (`qlora-primary-dpo.zip`, 47MB adapter only)
- HuggingFace Hub for base models (auto-downloaded on first run)
- `models/router.pkl` IS in the repo (small, 70.4% accuracy complexity classifier)

## Hardware requirements
| Component | Minimum | Recommended |
|---|---|---|
| Student model inference | 6GB VRAM | 8GB VRAM |
| Teacher model inference | 6GB VRAM | 8GB VRAM |
| Both simultaneously | Not supported on 6GB | 12GB+ VRAM |
| CPU fallback | 16GB RAM | 32GB RAM |

## Key files
- `inference/serve.py` -- HybridModelServer (router + student + teacher)
- `inference/router.py` -- ComplexityRouter (routes simple vs complex problems)
- `inference/export_gguf.py` -- export any checkpoint to GGUF for llama.cpp
- `docs/serve_api_contract.md` -- full API spec for serve.py
- `docs/cuda_vulkan_comparison.md` -- inference speed benchmarks

## Troubleshooting
- **OOM on model load:** reduce `max_seq_length` in `HybridModelServer` init
- **Router not found:** ensure `models/router.pkl` exists (committed to repo)
- **Slow teacher inference:** teacher loads lazily -- first call takes 30-60s
- **Docker sandbox fails:** ensure Docker Desktop is running before starting backend