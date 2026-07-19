"""
Builds train/val/test split (80/10/10) from passing trajectories.
Works on any passing-trajectories JSON file — currently pointed at Faiza's
subset alone; swap INPUT_FILE to the merged/deduped dataset once Sakshi's
collector.py output lands.
"""
import json
import random

INPUT_FILE = "datasets/raw/trajectories_faiza_14b_passing.json"
TRAIN_OUT = "datasets/splits/train.json"
VAL_OUT = "datasets/splits/val.json"
TEST_OUT = "datasets/splits/test.json"

SEED = 42


def main():
    with open(INPUT_FILE) as f:
        data = json.load(f)

    items = list(data.items())
    random.Random(SEED).shuffle(items)

    n = len(items)
    n_train = int(n * 0.8)
    n_val = int(n * 0.1)

    train = dict(items[:n_train])
    val = dict(items[n_train:n_train + n_val])
    test = dict(items[n_train + n_val:])

    import os
    os.makedirs("datasets/splits", exist_ok=True)

    with open(TRAIN_OUT, "w") as f:
        json.dump(train, f, indent=2)
    with open(VAL_OUT, "w") as f:
        json.dump(val, f, indent=2)
    with open(TEST_OUT, "w") as f:
        json.dump(test, f, indent=2)

    print(f"Total: {n}")
    print(f"Train: {len(train)} ({len(train)/n:.1%})")
    print(f"Val:   {len(val)} ({len(val)/n:.1%})")
    print(f"Test:  {len(test)} ({len(test)/n:.1%})")
    print("\nNOTE: target 800-1500 examples per plan — currently running on")
    print("Faiza's subset only (143). Swap INPUT_FILE to merged dataset")
    print("once Sakshi's collector.py output is available.")


if __name__ == "__main__":
    main()