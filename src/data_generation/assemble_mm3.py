"""Assemble mm3_data (multi-material train/test) from elastic 2048 data + newly
generated plasticine/sand h5, honoring the frozen split in configs/mm3_test_split.json.

Run on the server after uploading the new h5 dir. Uses symlinks by default (no copy).

Example:
  python assemble_mm3.py \
      --elastic_train /path/to/diff_E_2048_data/2048_data/2048_train \
      --elastic_test  /path/to/diff_E_2048_data/2048_data/2048_test \
      --new_h5        /path/to/uploaded/outputs_mpm/h5 \
      --out           /path/to/mm3_data \
      [--copy]
"""
import argparse
import json
import os
import shutil


def place(src, dst, copy):
    if os.path.lexists(dst):
        return
    if copy:
        shutil.copy2(src, dst)
    else:
        os.symlink(os.path.abspath(src), dst)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--elastic_train", required=True)
    ap.add_argument("--elastic_test", required=True)
    ap.add_argument("--new_h5", required=True, help="dir with *_001.h5 (plasticine) and *_002.h5 (sand)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--split", default=os.path.join(os.path.dirname(__file__), "..", "configs", "mm3_test_split.json"))
    ap.add_argument("--copy", action="store_true", help="copy files instead of symlinking")
    args = ap.parse_args()

    with open(args.split) as f:
        split = json.load(f)
    test_idx = {"001": set(split["plasticine_test"]), "002": set(split["sand_test"])}

    train_dir = os.path.join(args.out, "mm3_train")
    test_dir = os.path.join(args.out, "mm3_test")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    counts = {"train": {}, "test": {}}

    for src_dir, bucket in [(args.elastic_train, "train"), (args.elastic_test, "test")]:
        for name in sorted(os.listdir(src_dir)):
            if not name.endswith(".h5"):
                continue
            place(os.path.join(src_dir, name), os.path.join(train_dir if bucket == "train" else test_dir, name), args.copy)
            counts[bucket]["elastic"] = counts[bucket].get("elastic", 0) + 1

    for name in sorted(os.listdir(args.new_h5)):
        if not name.endswith(".h5"):
            continue
        idx_str, mat = name[:-3].split("_")
        if mat not in test_idx:
            raise ValueError(f"unexpected material code in {name}")
        bucket = "test" if int(idx_str) in test_idx[mat] else "train"
        place(os.path.join(args.new_h5, name), os.path.join(test_dir if bucket == "test" else train_dir, name), args.copy)
        key = "plasticine" if mat == "001" else "sand"
        counts[bucket][key] = counts[bucket].get(key, 0) + 1

    expected = {"train": split["train_counts_after_exclusion"],
                "test": {"elastic": 13, "plasticine": 14, "sand": 14}}
    print(f"mm3_train: {counts['train']}  (expected {expected['train']})")
    print(f"mm3_test:  {counts['test']}  (expected {expected['test']})")
    for bucket in ("train", "test"):
        assert counts[bucket] == expected[bucket], f"{bucket} counts mismatch — investigate before training"
    print("OK: counts match frozen split.")


if __name__ == "__main__":
    main()
