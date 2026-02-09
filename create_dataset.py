#!/usr/bin/env python3
"""
Create a simple 3D inpainting dataset by generating random cube masks for T1w volumes.

Expected input layout (BIDS-like):
  <input_root>/sub-*/ses-open/anat/*_T1w.nii.gz

Outputs (one-depth case folders under output_root):
  <output_root>/<CASE_ID>/
    - <CASE_ID>_T1w.nii.gz                    (only if --write-voided)
    - <CASE_ID>_mask-healthy(.nii.gz or -0000.nii.gz)
    - <CASE_ID>_voided(.nii.gz or -0000.nii.gz)       (only if --write-voided)

Plus:
  <output_root>/<manifest-name> (CSV listing paths)

Notes:
- Masks are axis-aligned cubes placed uniformly at random within volume bounds (no brain/tissue constraint).
- Affine + header are preserved from the input T1 when saving masks/voided.
"""

import csv
import argparse
from pathlib import Path

import numpy as np
import nibabel as nib
from tqdm import tqdm


def find_t1w_files(input_root: Path, session: str) -> list[Path]:
    pattern = f"sub-*/{session}/anat/*_T1w.nii.gz"
    return sorted(input_root.glob(pattern))


def strip_niigz(name: str) -> str:
    if name.endswith(".nii.gz"):
        return name[:-7]
    if name.endswith(".nii"):
        return name[:-4]
    return Path(name).stem


def cube_bbox(center: tuple[int, int, int], side: int, shape: tuple[int, int, int]) -> tuple[slice, slice, slice] | None:
    half_lo = side // 2
    half_hi = side - half_lo  # total = side

    x, y, z = center
    x0, x1 = x - half_lo, x + half_hi
    y0, y1 = y - half_lo, y + half_hi
    z0, z1 = z - half_lo, z + half_hi

    if x0 < 0 or y0 < 0 or z0 < 0:
        return None
    if x1 > shape[0] or y1 > shape[1] or z1 > shape[2]:
        return None

    return (slice(x0, x1), slice(y0, y1), slice(z0, z1))


def generate_one_cube_mask_uniform(
    shape: tuple[int, int, int],
    rng: np.random.Generator,
    side_min: int,
    side_max: int,
    max_tries: int,
) -> np.ndarray:
    for _ in range(max_tries):
        side = int(rng.integers(side_min, side_max + 1))

        center = (
            int(rng.integers(0, shape[0])),
            int(rng.integers(0, shape[1])),
            int(rng.integers(0, shape[2])),
        )

        bb = cube_bbox(center, side, shape)
        if bb is None:
            continue

        mask = np.zeros(shape, dtype=bool)
        mask[bb] = True
        return mask

    raise RuntimeError(
        f"Could not sample an in-bounds cube after {max_tries} tries. "
        f"Try lowering --mask-side-max or increasing --max-tries."
    )


def ensure_case_dir(output_root: Path, case_id: str) -> Path:
    d = output_root / case_id
    d.mkdir(parents=True, exist_ok=True)
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-root", type=Path, required=True)
    ap.add_argument("--output-root", type=Path, required=True)
    ap.add_argument("--session", type=str, default="ses-open")

    ap.add_argument("--samples-per-volume", type=int, default=1)
    ap.add_argument("--mask-side-min", type=int, default=17)
    ap.add_argument("--mask-side-max", type=int, default=31)
    ap.add_argument("--max-tries", type=int, default=400)

    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument(
        "--write-voided",
        action="store_true",
        help="If set: save original T1w + voided volumes inside each case folder.",
    )
    ap.add_argument("--manifest-name", type=str, default="pairs.csv")

    args = ap.parse_args()

    input_root = args.input_root
    output_root = args.output_root
    output_root.mkdir(parents=True, exist_ok=True)

    t1_files = find_t1w_files(input_root, args.session)
    if not t1_files:
        raise SystemExit(f"No T1w files found under {input_root} with session '{args.session}'.")

    rng_global = np.random.default_rng(args.seed)

    rows = []

    for t1_path in tqdm(t1_files, desc="Generating cube masks"):
        img = nib.load(str(t1_path))
        t1 = img.get_fdata(dtype=np.float32)
        shape = t1.shape

        case_id = strip_niigz(t1_path.name)  # e.g. sub-s003_ses-open_T1w
        case_dir = ensure_case_dir(output_root, case_id)

        # Optionally save/copy the original T1w into the case folder
        out_t1_path = ""
        if args.write_voided:
            out_t1_path = case_dir / f"{case_id}.nii.gz"  # keep same base name
            if not out_t1_path.exists():
                nib.save(nib.Nifti1Image(t1.astype(np.float32), affine=img.affine, header=img.header), str(out_t1_path))

        # Per-volume RNG (reproducible as long as file order + seed stay fixed)
        local_seed = int(rng_global.integers(0, 2**63 - 1))
        rng = np.random.default_rng(local_seed)

        for i in range(args.samples_per_volume):
            mask_name = (
                f"{case_id}_mask-healthy.nii.gz"
                if args.samples_per_volume == 1
                else f"{case_id}_mask-healthy-{i:04d}.nii.gz"
            )
            out_mask_path = case_dir / mask_name

            mask = generate_one_cube_mask_uniform(
                shape=shape,
                rng=rng,
                side_min=args.mask_side_min,
                side_max=args.mask_side_max,
                max_tries=args.max_tries,
            )

            nib.save(nib.Nifti1Image(mask.astype(np.uint8), affine=img.affine, header=img.header), str(out_mask_path))

            out_voided_path = ""
            if args.write_voided:
                void_name = (
                    f"{case_id}_voided.nii.gz"
                    if args.samples_per_volume == 1
                    else f"{case_id}_voided-{i:04d}.nii.gz"
                )
                out_voided_path = case_dir / void_name

                voided = t1.copy()
                voided[mask] = 0.0
                nib.save(nib.Nifti1Image(voided.astype(np.float32), affine=img.affine, header=img.header), str(out_voided_path))

            rows.append(
                {
                    "input_t1_path": str(t1_path),
                    "output_t1_path": str(out_t1_path) if out_t1_path else "",
                    "mask_path": str(out_mask_path),
                    "voided_path": str(out_voided_path) if out_voided_path else "",
                }
            )

    manifest_path = output_root / args.manifest_name
    with open(manifest_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["input_t1_path", "output_t1_path", "mask_path", "voided_path"])
        writer.writeheader()
        writer.writerows(rows)

    print("\nDone.")
    print(f"Found T1w volumes: {len(t1_files)}")
    print(f"Generated masks: {len(rows)}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
