#!/usr/bin/env bash
# Run all experiments needed for the poster: headline GLUE numbers + rank ablation.
# Total estimated wall-clock on M-series MPS: ~60-90 min.
set -e
cd "$(dirname "$0")/.."
source .venv/bin/activate

mkdir -p results/runs

echo "=== Headline runs: MRPC ==="
python code/train.py --task mrpc --method lora --rank 8 --epochs 3 --batch-size 32
python code/train.py --task mrpc --method full --epochs 3 --batch-size 32
python code/train.py --task mrpc --method head --epochs 3 --batch-size 32

echo "=== Headline runs: SST-2 (capped to ~1 epoch for speed) ==="
python code/train.py --task sst2 --method lora --rank 8 --epochs 1 --batch-size 32 --max-train-steps 2000
python code/train.py --task sst2 --method full --epochs 1 --batch-size 32 --max-train-steps 2000

echo "=== Rank ablation on MRPC (r in {1,2,4}; r=8 already done) ==="
python code/train.py --task mrpc --method lora --rank 1 --epochs 3 --batch-size 32
python code/train.py --task mrpc --method lora --rank 2 --epochs 3 --batch-size 32
python code/train.py --task mrpc --method lora --rank 4 --epochs 3 --batch-size 32

echo "=== All runs complete ==="
ls -la results/runs/
