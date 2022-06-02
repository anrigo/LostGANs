import os
from pathlib import Path
import cv2
import numpy as np
import torch
from tqdm import tqdm
from data.datasets import get_dataset
import matplotlib.pyplot as plt

device = torch.device(
    "cuda") if torch.cuda.is_available() else torch.device("cpu")


def depth_estimation(ds, mode, visualize=False, save=True, limit=None):
    '''
    Use MiDaS Large to estimate depth from each image in the dataset and save
    the depthmaps as .npy files
    '''

    # load dataset
    dataset = get_dataset(ds, None, mode, return_filenames=True)

    save_path = Path('datasets', ds + '-depth', mode)

    # create dir structure
    if save and not save_path.is_dir():
        os.makedirs(save_path)

    # Intel MiDaS Large
    model = torch.hub.load("intel-isl/MiDaS", 'DPT_Large')
    midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
    transform = midas_transforms.dpt_transform

    model.to(device)
    model.eval()

    if limit is None:
        limit = len(dataset)

    for index in tqdm(range(limit)):
        image, _, _, filename, flip = dataset[index]
        o_image = torch.clone(image)

        if not flip:
            # apply transforms to resize and normalize the image
            image = cv2.imread(str(Path(dataset.image_dir, filename)))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = transform(image).to(device)

            # predict depth
            with torch.no_grad():
                predicted_depth = model(image)

                # interpolate to original size
                prediction = torch.nn.functional.interpolate(
                    predicted_depth.unsqueeze(1),
                    size=o_image.shape[-2:],
                    mode="bicubic",
                    align_corners=False,
                )
                prediction = prediction.squeeze().cpu().numpy()

            # visualize
            if visualize:
                _, axs = plt.subplots(1, 2)
                axs[0].imshow(o_image.permute(1, 2, 0) * 0.5 + 0.5)
                axs[1].imshow(prediction, cmap='gray')
                plt.show()

            # save depthmap
            if save:
                np.save(Path(save_path, filename + '.npy'), prediction)


if __name__ == "__main__":
    ds = 'coco'
    mode = 'val'

    depth_estimation(ds, mode, visualize=True, save=False)
