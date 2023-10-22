import os

from tqdm import tqdm

jpg_paths = []
for root, _, files in os.walk("datasets/deep-fashion-test/img_highres/MEN"):
    for f in files:
        if f.lower().endswith(".jpg"):
            jpg_paths.append(os.path.join(root, f))
print("found", len(jpg_paths), "images")
for path in tqdm(jpg_paths):
    lucfrncs_path = path.replace(".jpg", "_lucfrncs-step00003000.png_00001_.png")
    if not os.path.exists(lucfrncs_path):
        print(lucfrncs_path)
