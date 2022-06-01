# %%
import os
from pathlib import Path
import numpy as np
import torch
from tqdm import tqdm
from data.datasets import get_dataset
import matplotlib.pyplot as plt
from transformers import GLPNFeatureExtractor, GLPNForDepthEstimation, DPTFeatureExtractor, DPTForDepthEstimation
# from PIL import Image


ds = 'coco'
mode = 'val'

data = get_dataset(ds, None, mode)

save_path = Path('datasets', ds + '-depth', mode)

# create dir structure
if not save_path.is_dir():
  os.makedirs(save_path)


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

# print(len(data))
# print(len(data.image_ids))


# %%
for index in tqdm(range(0, len(data))):
  image, _, _ = data[index]


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

  # print(prediction.shape)


  # visualize
  _, axs = plt.subplots(1, 2)
  axs[0].imshow(image.permute(1,2,0) *  0.5 + 0.5)
  axs[1].imshow(prediction, cmap='gray')
  # axs[2].imshow(formatted, cmap='gray')
  plt.show()


  # save depthmap
  # np.save(Path(save_path, str(index) + '.npy'), prediction)

# %%
