import os
from pathlib import Path
import numpy as np
import torch
from tqdm import tqdm
from data.datasets import get_dataset
import matplotlib.pyplot as plt
from transformers import GLPNFeatureExtractor, GLPNForDepthEstimation, DPTFeatureExtractor, DPTForDepthEstimation
# from PIL import Image

device = torch.device(
    "cuda") if torch.cuda.is_available() else torch.device("cpu")

ds = 'coco'
mode = 'train'

dataset = get_dataset(ds, None, mode, return_filenames=True)

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

model.to(device)


for index in tqdm(range(len(dataset))):
    image, _, _, filename, flip = dataset[index]

    # extract features
    pixel_values = feature_extractor(image, return_tensors="pt").pixel_values
    pixel_values.to(device)
    pixel_values = pixel_values.type(torch.cuda.FloatTensor)

    # predict depth
    with torch.no_grad():
        outputs = model(pixel_values)
        predicted_depth = outputs.predicted_depth

    # print(predicted_depth.shape)

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
    # _, axs = plt.subplots(1, 2)
    # axs[0].imshow(image.permute(1,2,0) *  0.5 + 0.5)
    # axs[1].imshow(prediction, cmap='gray')
    # # axs[2].imshow(formatted, cmap='gray')
    # plt.show()

    # save depthmap
    np.save(Path(save_path, filename +
            ('_flip' if flip else '') + '.npy'), prediction)
