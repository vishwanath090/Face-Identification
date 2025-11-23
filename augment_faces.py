from pathlib import Path
import cv2
import numpy as np
import albumentations as A
import random
import sys
import time

# === EDIT THESE FOLDERS IF NEEDED ===
FOLDERS = [
    r"C:\Users\vishw\OneDrive\Desktop\deep_learning\dataset\val",
    r"C:\Users\vishw\OneDrive\Desktop\deep_learning\dataset\test",
    r"C:\Users\vishw\OneDrive\Desktop\deep_learning\dataset\train",
]

AUGS_PER_IMAGE = 3                              # 3 augmentations for each original
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}

def build_pipeline():
    transforms = [
        A.HorizontalFlip(p=0.40),
        A.Affine(translate_percent={"x": (-0.02, 0.02), "y": (-0.02, 0.02)},
                 scale=(0.90, 1.10),
                 rotate=(-10, 10),
                 fit_output=False,
                 p=0.85),
        A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.60),
        A.HueSaturationValue(hue_shift_limit=6, sat_shift_limit=12, val_shift_limit=12, p=0.35),
        A.GaussianBlur(blur_limit=(3, 5), p=0.15),
        A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.01, 0.05), p=0.20),
    ]
    try:
        transforms.append(A.ImageCompression(quality_lower=85, quality_upper=95, p=0.30))
    except AttributeError:
        pass
    return A.Compose(transforms)

augment = build_pipeline()

def is_image(p: Path) -> bool:
    return p.suffix.lower() in IMG_EXTS and not p.name.startswith("aug_")

def main():
    print("=== Augmentation started ===")
    print(f"[INFO] Python: {sys.version.split()[0]}")
    print(f"[INFO] Albumentations: {getattr(A, '__version__', 'unknown')}")
    start_time = time.time()

    total_created = 0
    folder_count = 0

    for folder in FOLDERS:
        p = Path(folder)
        print(f"\n[CHECK] Folder: {p}")
        if not p.exists() or not p.is_dir():
            print("  [ERROR] Missing or not a directory. Skipping.")
            continue

        folder_count += 1
        imgs = sorted([f for f in p.iterdir() if f.is_file() and is_image(f)])
        print(f"  [INFO] Found {len(imgs)} originals with extensions: "
              f"{', '.join(sorted({x.suffix.lower() for x in imgs}) or {'(none)'})}")

        if not imgs:
            print("  [WARN] No images to augment here.")
            continue

        created_here = 0
        for img_path in imgs:
            img = cv2.imread(str(img_path))
            if img is None:
                print(f"  [WARN] Cannot read: {img_path.name}")
                continue

            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            base = img_path.stem

            for k in range(1, AUGS_PER_IMAGE + 1):
                aug_img = augment(image=rgb)["image"]
                out_bgr = cv2.cvtColor(aug_img, cv2.COLOR_RGB2BGR)

                out_path = p / f"aug_{base}_v{k}.jpg"
                c = 1
                while out_path.exists():
                    out_path = p / f"aug_{base}_v{k}_{c}.jpg"
                    c += 1

                ok = cv2.imwrite(str(out_path), out_bgr)
                if ok:
                    created_here += 1
                    total_created += 1
                    print(f"    [+] Wrote {out_path.name}")
                else:
                    print(f"    [ERROR] Failed to write {out_path.name}")

        print(f"  [DONE] {p.name}: created {created_here} augmented images.")

    elapsed = time.time() - start_time
    print("\n=== Augmentation finished ===")
    print(f"[SUMMARY] Folders processed: {folder_count}")
    print(f"[SUMMARY] Total augmented images created: {total_created}")
    print(f"[TIME] {elapsed:.2f} seconds")

if __name__ == "__main__":
    random.seed(42)
    np.random.seed(42)
    main()