#!/usr/bin/env bash
# Remaining runs after MRPC LoRA r=8 already completed.
cd "$(dirname "$0")/.."
. .venv/bin/activate

run() {
    echo "=== $* ==="
    python code/train.py "$@" || echo "FAILED: $*"
}

run --task mrpc --method full --epochs 3 --batch-size 32
run --task mrpc --method head --epochs 3 --batch-size 32
run --task mrpc --method lora --rank 1 --epochs 3 --batch-size 32
run --task mrpc --method lora --rank 2 --epochs 3 --batch-size 32
run --task mrpc --method lora --rank 4 --epochs 3 --batch-size 32
run --task sst2 --method lora --rank 8 --epochs 1 --batch-size 32 --max-train-steps 2000

echo "=== ALL DONE ==="
ls -la results/runs/
