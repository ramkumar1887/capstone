import cv2
import numpy as np
import os
from ultralytics import YOLO, SAM

# ── Config ──────────────────────────────────────────────────────────────────
BASE_DIR   = r"C:\Users\barag\PycharmProjects\AI\capstone\eggplant_dataset\diseases"
DISEASE    = "Fruit_rot"
OUTPUT_DIR = os.path.join(BASE_DIR, DISEASE + "_disease_labels")
os.makedirs(OUTPUT_DIR, exist_ok=True)
# ────────────────────────────────────────────────────────────────────────────

det_model     = YOLO("yolo26x.pt")        # Stage 1: detect & segment fruit
disease_model = YOLO("yolo26x.pt")   # Stage 2: detect disease (train this separately)
sam_model     = SAM("sam2_b.pt")          # Stage 2: segment disease area precisely

images_dir = os.path.join(BASE_DIR, DISEASE)
image_files = [f for f in os.listdir(images_dir)
               if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

for img_file in image_files:
    img_path = os.path.join(images_dir, img_file)
    img      = cv2.imread(img_path)
    h, w     = img.shape[:2]

    # ── STAGE 1: Segment the fruit ───────────────────────────────────────
    results = det_model(img_path, device="cuda", conf=0.25)[0]

    if results.masks is None:
        print(f"No fruit detected in {img_file}, skipping...")
        continue

    # Get the fruit mask (largest detected object = the fruit)
    masks   = results.masks.data.cpu().numpy()
    areas   = [m.sum() for m in masks]
    fruit_mask = masks[np.argmax(areas)]  # pick largest = fruit

    # Resize mask to image size
    fruit_mask = cv2.resize(fruit_mask, (w, h)).astype(np.uint8)

    # Remove background — keep only fruit pixels
    fruit_only = cv2.bitwise_and(img, img, mask=fruit_mask)

    # ── STAGE 2: Detect disease on fruit-only image ──────────────────────
    disease_results = disease_model(fruit_only, device="cuda", conf=0.20)[0]

    if len(disease_results.boxes) == 0:
        print(f"No disease detected in {img_file}, skipping...")
        continue

    # Use SAM to get precise disease masks from bounding boxes
    bboxes = disease_results.boxes.xyxy.cpu().numpy()
    sam_results = sam_model(img_path, bboxes=bboxes, device="cuda")[0]

    # ── Save labels ──────────────────────────────────────────────────────
    label_path = os.path.join(OUTPUT_DIR, os.path.splitext(img_file)[0] + ".txt")
    with open(label_path, "w") as f:
        if sam_results.masks is not None:
            for mask in sam_results.masks.data.cpu().numpy():
                mask_resized = cv2.resize(mask, (w, h)).astype(np.uint8)
                contours, _  = cv2.findContours(mask_resized, cv2.RETR_EXTERNAL,
                                                cv2.CHAIN_APPROX_SIMPLE)
                for contour in contours:
                    if len(contour) >= 3:
                        points = contour.reshape(-1, 2)
                        norm   = points / [w, h]   # normalize to 0-1
                        coords = " ".join(f"{x:.6f} {y:.6f}" for x, y in norm)
                        f.write(f"0 {coords}\n")   # class 0 = disease

    print(f"✓ {img_file} → labels saved")

print(f"\n✅ Done! Disease labels saved to: {OUTPUT_DIR}")