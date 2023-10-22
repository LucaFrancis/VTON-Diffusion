from PIL import Image
from lang_sam import LangSAM
from torchvision.transforms.functional import to_pil_image
import matplotlib.pyplot as plt
import numpy as np

model = LangSAM()
image_pil = Image.open("datasets/deep-fashion-test/img_highres/MEN/Shorts/id_00001859/01_1_front.jpg").convert("RGB")
#image_pil = Image.open("datasets/deep-fashion-test/img_highres/MEN/Denim/id_00000265/01_1_front.jpg").convert("RGB")
text_prompt = "clothing"
masks, boxes, phrases, logits = model.predict(image_pil, text_prompt)
#plt.imsave("datasets/deep-fashion-test/img_highres/MEN/Denim/id_00000265/01_1_front_lucfrncs-step00003000.png_00001_.png_mask_sam.png", arr=np.squeeze(masks))
print(masks, boxes, phrases, logits)
print(masks[0].shape)




