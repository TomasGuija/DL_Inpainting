import os
import numpy as np
import nibabel as nib


def normalize_image(image):
    min_val = np.min(image)
    max_val = np.max(image)
    # avoid div-by-zero if constant image
    denom = (max_val - min_val) if (max_val - min_val) != 0 else 1.0
    return (image - min_val) / denom


def crop_or_pad_image(image, target_shape):
    """
    Crop or pad the image to the target shape (center crop / center pad).
    """
    current_shape = image.shape
    cropped_padded_image = np.zeros(target_shape, dtype=image.dtype)

    crop_start = [
        (current_dim - target_dim) // 2 if current_dim > target_dim else 0
        for current_dim, target_dim in zip(current_shape, target_shape)
    ]
    crop_end = [crop_start[i] + target_shape[i] for i in range(len(target_shape))]

    pad_start = [
        (target_dim - current_dim) // 2 if current_dim < target_dim else 0
        for current_dim, target_dim in zip(current_shape, target_shape)
    ]
    pad_end = [pad_start[i] + min(current_shape[i], target_shape[i]) for i in range(len(target_shape))]

    cropped_image = image[
        crop_start[0]:crop_end[0],
        crop_start[1]:crop_end[1],
        crop_start[2]:crop_end[2],
    ]

    cropped_padded_image[
        pad_start[0]:pad_end[0],
        pad_start[1]:pad_end[1],
        pad_start[2]:pad_end[2],
    ] = cropped_image

    return cropped_padded_image


def process_images(input_folder, output_folder):
    """
    Input per case folder (your generator output):
      <case>/<case>.nii.gz                       (T1w)
      <case>/<case>_mask-healthy.nii.gz          (mask)

    Output per case folder (what BRATSDataset expects for training, test_flag=False):
      <case>/<case>-healthy.nii.gz               (normalized/cropped T1w)
      <case>/<case>-mask.nii.gz                  (cropped mask, kept binary-ish)
      <case>/<case>-diseased.nii.gz              (voided version of healthy)
    """
    target_shape = (224, 224, 224)
    os.makedirs(output_folder, exist_ok=True)

    for subdir, _, files in os.walk(input_folder):
        # Your current generator saves the original as "<case>.nii.gz" (case folder name == case id)
        # We only treat files that end exactly with ".nii.gz" and are NOT masks/voided
        t1_candidates = [
            f for f in files
            if f.endswith(".nii.gz")
            and "_mask-healthy" not in f
            and "_voided" not in f
            and "-mask" not in f
            and "-healthy" not in f
            and "-diseased" not in f
        ]

        for t1_file in t1_candidates:
            t1_path = os.path.join(subdir, t1_file)
            base = t1_file[:-7]  # strip ".nii.gz"

            # Your mask naming (from cube generator): "<case>_mask-healthy.nii.gz"
            mask_in = os.path.join(subdir, f"{base}_mask-healthy.nii.gz")
            if not os.path.exists(mask_in):
                # If it doesn't exist, skip this sample folder
                continue

            # Mirror one-depth structure in output folder
            relative_subdir = os.path.relpath(subdir, input_folder)
            output_subdir = os.path.join(output_folder, relative_subdir)
            os.makedirs(output_subdir, exist_ok=True)

            # Load images
            t1_img = nib.load(t1_path)
            t1 = t1_img.get_fdata().astype(np.float32)
            mask = nib.load(mask_in).get_fdata().astype(np.float32)

            # T1: clip + normalize
            t1 = np.clip(t1, np.quantile(t1, 0.001), np.quantile(t1, 0.999))
            t1 = normalize_image(t1)

            # Mask: ensure binary-ish BEFORE padding/cropping (so padding stays 0)
            mask = (mask > 0.5).astype(np.float32)

            # Crop/pad to 224^3
            t1 = crop_or_pad_image(t1, target_shape).astype(np.float32)
            mask = crop_or_pad_image(mask, target_shape).astype(np.float32)

            # Create diseased (voided) from healthy + mask
            diseased = t1.copy()
            diseased[mask > 0.5] = 0.0

            # Save with identity affine (to match their pipeline expectations)
            affine = np.eye(4)

            out_healthy = os.path.join(output_subdir, f"{base}-healthy.nii.gz")
            out_mask = os.path.join(output_subdir, f"{base}-mask.nii.gz")
            out_diseased = os.path.join(output_subdir, f"{base}-diseased.nii.gz")

            nib.save(nib.Nifti1Image(t1, affine), out_healthy)
            nib.save(nib.Nifti1Image(mask, affine), out_mask)
            nib.save(nib.Nifti1Image(diseased, affine), out_diseased)

            # Optional: print progress per sample folder
            # print("Saved:", out_healthy, out_mask, out_diseased)


if __name__ == "__main__":
    input_folder = r"Z:\\DATASETS\\2026_Local_Inpainting_T1w"
    output_folder = r"Z:\\DATASETS\\2026_Local_Inpainting_T1w_preprocessed"
    process_images(input_folder, output_folder)
