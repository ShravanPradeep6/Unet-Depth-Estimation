from pathlib import Path
import argparse
import os
import shutil

RGB_EXTS = {".jpg", ".jpeg"}
GT_EXTS = {".png"}

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
    ap = argparse.ArgumentParser("Split RGB JPGs and depth PNGs that share the same stem into two folders.")
    ap.add_argument("--in", dest="in_dir", required=True, help="Root folder to scan (recursively).")
    ap.add_argument("--out", required=True, help="Output root folder (creates images/ and gt/).")
    ap.add_argument("--mode", choices=["copy", "symlink", "hardlink"], default="copy",
                    help="copy is safest; symlink saves space.")
    ap.add_argument("--digits", type=int, default=6, help="Zero-pad output names (default 6).")
    ap.add_argument("--start-index", type=int, default=0, help="Start index for output naming.")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    in_root = Path(args.in_dir)
    out_root = Path(args.out)
    out_img = out_root / "images"
    out_gt = out_root / "gt"
    out_img.mkdir(parents=True, exist_ok=True)
    out_gt.mkdir(parents=True, exist_ok=True)

    # Build maps: stem -> path
    rgbs = {}
    gts = {}

    for p in in_root.rglob("*"):
        if not p.is_file():
            continue
        suf = p.suffix.lower()
        stem = p.stem  # e.g., "1" from "1.jpg"
        if suf in RGB_EXTS:
            # If duplicates exist, last one wins; you can change this if needed.
            rgbs[stem] = p
        elif suf in GT_EXTS:
            gts[stem] = p

    common = sorted(set(rgbs.keys()) & set(gts.keys()), key=lambda s: (len(s), s))
    if not common:
        print("No matching (stem.jpg, stem.png) pairs found.")
        return

    idx = args.start_index
    kept = 0

    for stem in common:
        rgb = rgbs[stem]
        gt = gts[stem]

        out_stem = str(idx).zfill(args.digits)
        rgb_dst = out_img / f"{out_stem}{rgb.suffix.lower()}"
        gt_dst  = out_gt  / f"{out_stem}{gt.suffix.lower()}"

        if args.dry_run:
            print(f"[pair {out_stem}] {rgb} -> {rgb_dst}")
            print(f"[pair {out_stem}] {gt}  -> {gt_dst}")
        else:
            link_or_copy(rgb, rgb_dst, args.mode)
            link_or_copy(gt,  gt_dst,  args.mode)

        idx += 1
        kept += 1

    print(f"Done. Wrote {kept} pairs.")
    print(f"images: {out_img}")
    print(f"gt:     {out_gt}")
    print(f"Next start index: {idx}")

if __name__ == "__main__":
    main()