# Changelog

## Week 1 — Setup & Environment
- Repo scaffolding, per-member environment setup (GPU/CUDA verification,
  Ollama teacher models, conda/venv environments)
- Docker-based sandbox executor for isolated code execution
- Tester agent wired to sandbox

## Week 2 — Full Teacher Pipeline
- 14B teacher model integrated into the LangGraph pipeline (Planner →
  Coder → Tester → Debugger)
- Docker Compose service added for sandbox builds
- 7B vs 14B teacher comparison and documented tradeoffs

## Week 3 — Trajectory Collection at Scale
- Full 179-task trajectory collection with 14B teacher (143/179 passed)
- Three real pipeline bugs found and fixed: debugger not stripping code
  fences, HumanEval `check()` never invoked (silently passing broken
  code), intermittent Ollama 500 errors (added retry logic)
- DPO preference pairs collected (rejected/preferred code pairs from
  retry attempts)

## Week 4 — Dataset Construction & Compression
- All three team members' trajectories merged and deduplicated (358
  total, 174 passing)
- Train/val/test split built and validated for balance across source
  and difficulty (139/17/18)
- Chain-of-thought compression pipeline for SFT-ready data

## Week 5 — QLoRA Fine-Tuning, Part 1
- Primary QLoRA training launched (Qwen2.5-Coder-7B, 4-bit, rank 32)
- Epoch 1: eval_loss 0.6163, val pass@1 88.2%
- Epoch 2: eval_loss 0.5461
- W&B tracking integrated across all team runs

## Week 6 — QLoRA Fine-Tuning, Part 2 + Export
- Epochs 3-4 completed; final eval_loss 0.495
- pass@1 plateaued at 87.5% from epoch 3 onward — diagnosed as a
  data-size ceiling (174 examples), not a config issue, confirmed by
  three independently converging configs (rank-32, rank-8, Llama3.1-8B)
- Best checkpoint (epoch 4) selected and exported
- Fixed a `datasets/__init__.py` shadowing bug affecting anyone
  importing unsloth from repo root

## Week 7 — Alignment (DPO) + Hybrid Routing
- DPO preference alignment run using 97 preference pairs
- Note: run on the rank-8 ablation checkpoint rather than the primary
  rank-32 model, due to a Windows Smart App Control policy blocking
  native ML library DLLs on the primary training laptop mid-week (a
  device-level issue, not resolvable via package changes or WSL2 in
  time) — documented transparently in `docs/dpo_alignment_results.md`
- rewards/accuracies improved 0.225 → 0.588 during alignment

## Week 8 — Writeup & Release
- Backend (FastAPI), frontend (Streamlit demo), Docker Compose stack
  for full-service deployment
- API reference, deployment guide, and troubleshooting docs
- Model card, final benchmarks, and README badges
- GGUF quantization benchmarking (CUDA vs Vulkan)
- Final report and this changelog

## Team
- Yeshita Motwani
- Faiza Bagban
- Sakshi Kolhe
