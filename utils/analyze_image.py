import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

from tqdm import tqdm

#image = Image.open("/home/luca/ComfyUI/output/transform_all/26.09.2023-17:59:30/home/luca/virtual-tryon-diffusion/datasets/deep-fashion-test/img_highres/MEN/Sweaters/id_00006356/01_1_front_fabhsr-step00002500.png_mask.png_00001_.png")
#image = Image.open("datasets/deep-fashion-test/img_highres/MEN/Denim/id_00000265/01_1_front_mask.png")
image = Image.open("datasets/deep-fashion-test/img_highres/MEN/Denim/id_00000265/01_1_front_mask_sam.png")
#image = Image.open("/home/luca/Downloads/ComfyUI_temp_kggpo_00002_.png")
image = np.asarray(image)
fig, ax = plt.subplots()
values = []
for i in tqdm(range(image.shape[0])):
    for j in range(image.shape[1]):
        if not np.isin(image[i, j], values).any():
            print(image[i, j])
            values.append(image[i, j])
