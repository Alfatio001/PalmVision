from ultralytics import YOLO
import cv2
import os
import csv
from statistics import mean

# ==========================================================
# CONFIGURATION
# ==========================================================

MODEL_PATH = "models/best.pt"

INPUT_FOLDER = "test/images"

OUTPUT_FOLDER = "output"

CSV_FILE = "tree_analysis.csv"

CONFIDENCE = 0.25

# ==========================================================
# LOAD MODEL
# ==========================================================

model = YOLO(MODEL_PATH)

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# ==========================================================
# CREATE CSV
# ==========================================================

with open(CSV_FILE, mode="w", newline="") as file:

    writer = csv.writer(file)

    writer.writerow([
        "Image",
        "Tree_ID",
        "Confidence",
        "Center_X",
        "Center_Y",
        "Width",
        "Height",
        "Area",
        "Canopy_Size"
    ])

    total_images = 0
    total_trees = 0

    all_width = []
    all_height = []
    all_area = []
    all_canopy = []

    image_extensions = (".jpg", ".jpeg", ".png", ".bmp")

    images = sorted([
        f for f in os.listdir(INPUT_FOLDER)
        if f.lower().endswith(image_extensions)
    ])

    print("=" * 60)
    print(f"Found {len(images)} images")
    print("=" * 60)

    for filename in images:

        total_images += 1

        image_path = os.path.join(INPUT_FOLDER, filename)

        print(f"\nProcessing : {filename}")

        results = model.predict(
            source=image_path,
            conf=CONFIDENCE,
            save=False,
            verbose=False
        )

        result = results[0]

        annotated = result.plot()

        output_path = os.path.join(
            OUTPUT_FOLDER,
            filename
        )

        cv2.imwrite(output_path, annotated)

        boxes = result.boxes

        tree_count = len(boxes)

        total_trees += tree_count

        print(f"Detected Trees : {tree_count}")

        if tree_count == 0:
            continue

        widths = []

        heights = []

        areas = []

        canopies = []

        for idx, box in enumerate(boxes):

            x1, y1, x2, y2 = box.xyxy[0].tolist()

            confidence = float(box.conf[0])

            width = x2 - x1

            height = y2 - y1

            area = width * height

            canopy_size = (width + height) / 2

            center_x = (x1 + x2) / 2

            center_y = (y1 + y2) / 2

            widths.append(width)

            heights.append(height)

            areas.append(area)

            canopies.append(canopy_size)

            all_width.append(width)

            all_height.append(height)

            all_area.append(area)

            all_canopy.append(canopy_size)

            writer.writerow([
                filename,
                idx + 1,
                round(confidence,3),
                round(center_x,2),
                round(center_y,2),
                round(width,2),
                round(height,2),
                round(area,2),
                round(canopy_size,2)
            ])

        print(f"Average Width  : {mean(widths):.2f} px")
        print(f"Average Height : {mean(heights):.2f} px")
        print(f"Average Area   : {mean(areas):.2f} px²")
        print(f"Average Canopy Size : {mean(canopies):.2f} px")

print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)

print(f"Total Images : {total_images}")
print(f"Total Trees  : {total_trees}")

if len(all_width) > 0:

    print(f"Average Width  : {mean(all_width):.2f} px")
    print(f"Average Height : {mean(all_height):.2f} px")
    print(f"Average Area   : {mean(all_area):.2f} px²")
    print(f"Average Canopy Size : {mean(all_canopy):.2f} px")

print("\nDetection Finished Successfully!")
print(f"Annotated Images : {OUTPUT_FOLDER}")
print(f"CSV File         : {CSV_FILE}")