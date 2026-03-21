#MADE THIS TO MAKE USE OF DATASETS THAT HAVE ALTERNATING RGB/GT IMAGE
#CODE CRASHES IF IT ALREADY FINDS AN IMAGE IT'S TRYING TO ADD IN THE FOLDER

from pathlib import Path
import argparse
import os
import shutil

IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}

def is_image(p: Path) -> bool:
    return p.suffix.lower() in IMG_EXTS and p.is_file()

def link_or_copy(src: Path, dst: Path, mode: str):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        raise FileExistsError(f"Refusing to overwrite existing file: {dst}")

    if mode == "copy":
        shutil.copy2(src, dst)
    elif mode == "symlink":
        # relative symlink when possible
        rel = os.path.relpath(src, start=dst.parent)
        os.symlink(rel, dst)
    elif mode == "hardlink":
        os.link(src, dst)
    else:
        raise ValueError(f"Unknown mode: {mode}")

def main():
    ap = argparse.ArgumentParser(description="Split alternating RGB/GT files into two folders.")
    ap.add_argument("--in", dest="inputs", nargs="+", required=True,
                    help="One or more input directories (e.g., basement_0001a_out basement_0002a_out)")
    ap.add_argument("--out", required=True, help="Output root directory (will create images/ and gt/ inside)")
    ap.add_argument("--rgb-first", action="store_true",
                    help="If set, assumes order is RGB,GT,RGB,GT,... (default).")
    ap.add_argument("--gt-first", action="store_true",
                    help="If set, assumes order is GT,RGB,GT,RGB,...")
    ap.add_argument("--start-index", type=int, default=0,
                    help="Starting integer index for output filenames (default 0).")
    ap.add_argument("--digits", type=int, default=6,
                    help="Zero-padding digits for output filenames (default 6 -> 000123).")
    ap.add_argument("--mode", choices=["copy", "symlink", "hardlink"], default="copy",
                    help="How to place files in output (default copy). symlink saves space.")
    ap.add_argument("--dry-run", action="store_true", help="Print actions without writing files.")
    args = ap.parse_args()

    if args.gt_first and args.rgb_first:
        raise SystemExit("Choose only one of --rgb-first or --gt-first.")
    rgb_first = True
    if args.gt_first:
        rgb_first = False

    out_root = Path(args.out)
    out_img = out_root / "images"
    out_gt = out_root / "gt"
    out_img.mkdir(parents=True, exist_ok=True)
    out_gt.mkdir(parents=True, exist_ok=True)

    idx = args.start_index
    total_pairs = 0

    for d in args.inputs:
        in_dir = Path(d)
        if not in_dir.is_dir():
            raise SystemExit(f"Not a directory: {in_dir}")

        files = sorted([p for p in in_dir.iterdir() if is_image(p)])
        if len(files) < 2:
            print(f"[skip] {in_dir} has <2 image files")
            continue
        if len(files) % 2 != 0:
            print(f"[warn] {in_dir} has odd count ({len(files)}). Last file will be ignored.")

        usable = files[: (len(files) // 2) * 2]  # even length
        # Pair by alternating position
        for i in range(0, len(usable), 2):
            a = usable[i]
            b = usable[i + 1]
            rgb = a if rgb_first else b
            gt  = b if rgb_first else a

            stem = str(idx).zfill(args.digits)
            rgb_dst = out_img / f"{stem}{rgb.suffix.lower()}"
            gt_dst  = out_gt  / f"{stem}{gt.suffix.lower()}"

            if args.dry_run:
                print(f"{rgb} -> {rgb_dst}")
                print(f"{gt}  -> {gt_dst}")
            else:
                link_or_copy(rgb, rgb_dst, args.mode)
                link_or_copy(gt,  gt_dst,  args.mode)

            idx += 1
            total_pairs += 1

    print(f"Done. Wrote {total_pairs} pairs to:")
    print(f"  {out_img}")
    print(f"  {out_gt}")
    print(f"Next start index would be {idx}.")

if __name__ == "__main__":
    main()