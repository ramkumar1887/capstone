import cv2
import numpy as np
import os
import random

# ── Config ──────────────────────────────────────────────────────────────────
BASE_DIR = r"C:\Users\barag\PycharmProjects\AI\capstone\eggplant_dataset\diseases"
DISEASE  = "Fruit_rot"   # ← change as needed
# ────────────────────────────────────────────────────────────────────────────

images_dir = os.path.join(BASE_DIR, DISEASE)
labels_dir = os.path.join(BASE_DIR, DISEASE + "_labels")

def draw_masks(image_path, label_path):
    img = cv2.imread(image_path)
    h, w = img.shape[:2]

    with open(label_path, "r") as f:
        lines = f.readlines()

    for line in lines:
        vals      = line.strip().split()
        class_id  = int(vals[0])
        points    = list(map(float, vals[1:]))

        # Convert normalized coords → pixel coords
        coords = [(int(points[i] * w), int(points[i+1] * h))
                  for i in range(0, len(points), 2)]

        # Random color per instance
        color = [random.randint(100, 255) for _ in range(3)]

        # Draw filled mask (semi-transparent)
        overlay  = img.copy()
        pts      = np.array(coords, dtype=np.int32)
        cv2.fillPoly(overlay, [pts], color)
        img      = cv2.addWeighted(overlay, 0.4, img, 0.6, 0)

        # Draw outline
        cv2.polylines(img, [pts], isClosed=True, color=color, thickness=2)

        # Draw class label
        cv2.putText(img, f"Class {class_id}", coords[0],
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    return img


# ── View all annotated images (press any key to go next, ESC to quit) ───────
image_files = [f for f in os.listdir(images_dir)
               if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

for img_file in image_files:
    label_file = os.path.splitext(img_file)[0] + ".txt"
    label_path = os.path.join(labels_dir, label_file)

    if not os.path.exists(label_path):
        print(f"No label for {img_file}, skipping...")
        continue

    image_path = os.path.join(images_dir, img_file)
    result     = draw_masks(image_path, label_path)

    cv2.imshow(f"Annotations - {img_file}", result)
    key = cv2.waitKey(0)

    cv2.destroyAllWindows()
    if key == 27:   # ESC to quit early
        break

print("Done viewing!")