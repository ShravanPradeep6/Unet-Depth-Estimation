#(MOST RECENT to automate dataset splitting)
from pathlib import Path
import argparse
import os
import shutil

def link_or_copy(src: Path, dst: Path, mode: str):
    dst.parent.mkdir(parents=True, exist_ok=True)

    if dst.exists():
        raise FileExistsError(f"Refusing to overwrite existing file: {dst}")

    if mode == "copy":
        shutil.copy2(src, dst)
    elif mode == "symlink":
        rel = os.path.relpath(src, start=dst.parent)
        os.symlink(rel, dst)
    elif mode == "hardlink":
        os.link(src, dst)
    else:
        raise ValueError(f"Unknown mode: {mode}")

def main():
    ap = argparse.ArgumentParser(
        description="Recursively collect JPG RGB images and PNG GT images from subfolders into images/ and gt/."
    )
    ap.add_argument("--root", required=True, help="Root directory containing many subfolders")
    ap.add_argument("--out", required=True, help="Output root directory (creates images/ and gt/ inside)")
    ap.add_argument("--start-index", type=int, default=0, help="Starting integer index")
    ap.add_argument("--digits", type=int, default=6, help="Zero-padding digits")
    ap.add_argument("--mode", choices=["copy", "symlink", "hardlink"], default="copy",
                    help="How to place files in output")
    ap.add_argument("--dry-run", action="store_true", help="Print actions without writing files")
    args = ap.parse_args()

    root = Path(args.root)
    if not root.is_dir():
        raise SystemExit(f"Not a directory: {root}")

    out_root = Path(args.out)
    out_img = out_root / "images"
    out_gt = out_root / "gt"
    out_img.mkdir(parents=True, exist_ok=True)
    out_gt.mkdir(parents=True, exist_ok=True)

    idx = args.start_index
    total_pairs = 0

    subdirs = sorted([p for p in root.rglob("*") if p.is_dir()])

    for d in subdirs:
        jpgs = sorted([p for p in d.iterdir() if p.is_file() and p.suffix.lower() == ".jpg"])
        pngs = sorted([p for p in d.iterdir() if p.is_file() and p.suffix.lower() == ".png"])

        if not jpgs and not pngs:
            continue

        if len(jpgs) != len(pngs):
            print(f"[warn] {d} has {len(jpgs)} JPGs and {len(pngs)} PNGs; pairing only up to min count.")

        pair_count = min(len(jpgs), len(pngs))
        if pair_count == 0:
            continue

        print(f"[processing] {d} ({pair_count} pairs)")

        for i in range(pair_count):
            rgb = jpgs[i]
            gt = pngs[i]

            stem = str(idx).zfill(args.digits)
            rgb_dst = out_img / f"{stem}.jpg"
            gt_dst = out_gt / f"{stem}.png"

            if args.dry_run:
                print(f"{rgb} -> {rgb_dst}")
                print(f"{gt}  -> {gt_dst}")
            else:
                link_or_copy(rgb, rgb_dst, args.mode)
                link_or_copy(gt, gt_dst, args.mode)

            idx += 1
            total_pairs += 1

    print(f"\nDone. Wrote {total_pairs} pairs to:")
    print(f"  {out_img}")
    print(f"  {out_gt}")
    print(f"Next start index would be {idx}.")

if __name__ == "__main__":
    main()