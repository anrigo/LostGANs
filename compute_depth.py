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
bs = 10

data = get_dataset(ds, None, mode)

# dataloader = torch.utils.data.DataLoader(
#                     data, batch_size=bs,
#                     drop_last=True, shuffle=False, num_workers=1)

save_path = Path('datasets', ds + '-depth', mode)

# create dir structure
if not save_path.is_dir():
  os.makedirs(save_path)

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


# for index, batch in tqdm(enumerate(dataloader)):
#   image, _, _ = batch

#   print(image.shape)

# for index, batch in tqdm(enumerate(dataloader)):
  # image, _, _ = batch


image = [data[i][0] for i in range(11)]


# image_id = data.image_ids[index]
# filename = data.image_id_to_filename[image_id]
# print(filename)


# extract features
pixel_values = feature_extractor(image[:], return_tensors="pt").pixel_values


# predict depth
with torch.no_grad():
  outputs = model(pixel_values)
  predicted_depth = outputs.predicted_depth

# interpolate to original size
predictions = []

for i, pred_depth in enumerate(predicted_depth):
  pred = torch.nn.functional.interpolate(
                      pred_depth.unsqueeze(0).unsqueeze(0),
                      size=image[i].shape[-2:],
                      mode="bicubic",
                      align_corners=False,
              )
  pred = pred.squeeze().cpu().numpy()
  predictions.append(pred)

# formatted = (prediction * 255 / np.max(prediction)).astype("uint8")
# formatted = Image.fromarray(formatted)

# print(prediction.shape)

# save depthmap

# np.save(Path(save_path, str(index) + '.npy'), prediction)
