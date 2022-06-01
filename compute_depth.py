# %%
import numpy as np
import torch
from data.datasets import get_dataset
import matplotlib.pyplot as plt
from transformers import GLPNFeatureExtractor, GLPNForDepthEstimation, DPTFeatureExtractor, DPTForDepthEstimation
from PIL import Image


data = get_dataset('coco', 640, 'test')


# %%
# load models

# NYU GLPN
# feature_extractor = GLPNFeatureExtractor.from_pretrained("vinvino02/glpn-nyu")
# model = GLPNForDepthEstimation.from_pretrained("vinvino02/glpn-nyu")

# KITTI GLPN
# feature_extractor = GLPNFeatureExtractor.from_pretrained("vinvino02/glpn-kitti")
# model = GLPNForDepthEstimation.from_pretrained("vinvino02/glpn-kitti")

# Intel DPT Large
feature_extractor = DPTFeatureExtractor.from_pretrained("Intel/dpt-large")
model = DPTForDepthEstimation.from_pretrained("Intel/dpt-large")


# %%
index = 5698
image, label, bbox = data[index]


# image_id = data.image_ids[index]
# filename = data.image_id_to_filename[image_id]
# print(filename)


# extract features
pixel_values = feature_extractor(image, return_tensors="pt").pixel_values


# predict depth
with torch.no_grad():
  outputs = model(pixel_values)
  predicted_depth = outputs.predicted_depth

print(predicted_depth.shape)

# interpolate to original size
prediction = torch.nn.functional.interpolate(
                    predicted_depth.unsqueeze(1),
                    size=image.shape[-2:],
                    mode="bicubic",
                    align_corners=False,
             )
prediction = prediction.squeeze().cpu().numpy()

# formatted = (prediction * 255 / np.max(prediction)).astype("uint8")
# formatted = Image.fromarray(formatted)

print(prediction.shape)


# visualize
_, axs = plt.subplots(1, 2)
axs[0].imshow(image.permute(1,2,0) *  0.5 + 0.5)
axs[1].imshow(prediction, cmap='gray')
# axs[2].imshow(formatted, cmap='gray')
plt.show()
