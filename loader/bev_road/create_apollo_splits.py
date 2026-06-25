"""
create_apollo_splits.py — Apollo Sim 3D Lane Dataset split Generator
================================================================

Source:   Apollo_Sim_3D_Lane_Release/laneline_label.json  (7498 line JSONL)
Output:   Apollo_Sim_3D_Lane_Release/splits/{standard|rare_subset|illus_chg}/
              train.json   (JSONL)
              val.json     (JSONL)

Definition of each split
------------------
standard    : Random 80/20 split of the entire dataset
rare_subset : train = same with standard 
              val   = only /06/ ~ /11/ folders of standard validation set   
illus_chg   : train = exclude /00/,/01/,/06/,/07/ 
              val   = /00/,/01/,/06/,/07/ 

------
python create_apollo_splits.py
python create_apollo_splits.py --seed 42 --batch_size 8
python create_apollo_splits.py --src /path/to/laneline_label.json
"""

import os
import random
import math
import argparse
import json


RARE_FOLDERS     = ['/06/', '/07/', '/08/', '/09/', '/10/', '/11/']
ILLUS_CHG_FOLDERS = ['/00/', '/01/', '/06/', '/07/']


def line_in_folders(line: str, folders: list) -> bool:
    return any(f in line for f in folders)


def align_to_batch(lines: list, batch_size: int) -> list:
    n = (len(lines) // batch_size) * batch_size
    return lines[:n]


def write_jsonl(path: str, lines: list):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        for line in lines:
            f.write(line if line.endswith('\n') else line + '\n')
    print(f"  saved {len(lines):5d} lines → {path}")


def create_splits(src: str, out_root: str, seed: int, batch_size: int):
    assert os.path.exists(src), f"Source not found: {src}"

    with open(src) as f:
        all_lines = f.readlines()
    total = len(all_lines)
    print(f"Total samples: {total}")

    # -----------------------------------------------------------------------
    # 1. standard — random 80/20 split
    # -----------------------------------------------------------------------
    rng = random.Random(seed)
    shuffled = all_lines[:]
    rng.shuffle(shuffled)

    n_train = int(math.floor(total * 0.8 / batch_size) * batch_size)
    n_val   = int(math.floor(total * 0.2 / batch_size) * batch_size)

    std_train = shuffled[:n_train]
    std_val   = shuffled[n_train:n_train + n_val]

    print("\n[standard]")
    write_jsonl(f"{out_root}/standard/train.json", std_train)
    write_jsonl(f"{out_root}/standard/val.json",   std_val)

    # -----------------------------------------------------------------------
    # 2. rare_subset — train = standard, val = 06~11
    # -----------------------------------------------------------------------
    rare_val = [l for l in all_lines if line_in_folders(l, RARE_FOLDERS)]
    rare_val = align_to_batch(rare_val, batch_size)

    print("\n[rare_subset]")
    write_jsonl(f"{out_root}/rare_subset/train.json", std_train)   # same as standard
    write_jsonl(f"{out_root}/rare_subset/val.json",   rare_val)

    # -----------------------------------------------------------------------
    # 3. illus_chg — train = exclude 00,01,06,07; val = 00,01,06,07
    # -----------------------------------------------------------------------
    illus_train_src = [l for l in all_lines
                       if not line_in_folders(l, ILLUS_CHG_FOLDERS)]
    illus_train = align_to_batch(illus_train_src, batch_size)

    illus_val_src = [l for l in all_lines
                     if line_in_folders(l, ILLUS_CHG_FOLDERS)]
    illus_val = align_to_batch(illus_val_src, batch_size)

    print("\n[illus_chg]")
    write_jsonl(f"{out_root}/illus_chg/train.json", illus_train)
    write_jsonl(f"{out_root}/illus_chg/val.json",   illus_val)

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    print("\n========= Split Summary =========")
    print(f"Source                : {src}  ({total} frames)")
    print(f"standard   train/val  : {len(std_train)} / {len(std_val)}")
    print(f"rare_subset train/val : {len(std_train)} / {len(rare_val)}")
    print(f"illus_chg  train/val  : {len(illus_train)} / {len(illus_val)}")
    print(f"Output root           : {out_root}")
    print("=================================\n")

    for split_name, train_lines, val_lines in [
        ('standard',    std_train,   std_val),
        ('rare_subset', std_train,   rare_val),
        ('illus_chg',   illus_train, illus_val),
    ]:
        print(f"[{split_name}] folder distribution (val):")
        counts = {}
        for l in val_lines:
            for i in range(12):
                tag = f'/{i:02d}/'
                if tag in l:
                    counts[tag] = counts.get(tag, 0) + 1
                    break
        for k in sorted(counts):
            print(f"  {k} : {counts[k]}")


def main():
    APOLLO_ROOT = "../../../dataset/Apollo_Sim_3D_Lane_Release"

    parser = argparse.ArgumentParser()
    parser.add_argument('--src', default=f"{APOLLO_ROOT}/laneline_label.json",
                        help='Source JSONL file (laneline_label.json)')
    parser.add_argument('--out', default=f"{APOLLO_ROOT}/splits",
                        help='Output root directory for splits')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for standard split shuffle')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch alignment size')
    args = parser.parse_args()

    create_splits(
        src=args.src,
        out_root=args.out,
        seed=args.seed,
        batch_size=args.batch_size,
    )


if __name__ == '__main__':
    main()
