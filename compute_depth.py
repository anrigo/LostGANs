import os
from pathlib import Path
import cv2
import numpy as np
import torch
from tqdm import tqdm
from data.datasets import get_dataset
import matplotlib.pyplot as plt
from torchvision.utils import draw_bounding_boxes

device = torch.device(
    "cuda") if torch.cuda.is_available() else torch.device("cpu")


def depth_estimation(ds, mode, visualize=True, save=False, limit=None):
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


def normalize(tensor: torch.Tensor, range: 'tuple[float, float]') -> torch.Tensor:
    '''Normalize tensor to the given range'''
    min_, max_ = range
    return min_ + (max_ - min_) * tensor


def scale_boxes(boxes: torch.Tensor, shape: 'tuple[int, int]') -> torch.Tensor:
    '''
    Scales bounding boxes to match the given image size

    Args:
        boxes: Tensor of bounding boxes in the (x, y, w, h) format, in the range (0,1)
        shape: tuple (height, width)
    Returns:
        boxes: Tensor of bounding boxes in the (xmin, ymin, xmax, ymax) format, scaled up to the specified shape
    '''
    for i, box in enumerate(boxes):
        x, y, w, h = box
        hh, ww = shape
        boxes[i] = torch.tensor(
            (int(x*ww), int(y*hh), int(x*ww)+int(w*ww), int(y*hh)+int(h*hh)))
    return boxes


if __name__ == "__main__":
    ds = 'coco'
    mode = 'val'
    limit = 1

    # load dataset
    dataset = get_dataset(ds, None, mode, return_filenames=True)

    if limit is None:
        limit = len(dataset)

    for index in range(limit):
        image, objs, boxes, filename, flip = dataset[index]
        boxes = torch.from_numpy(boxes)

        depthmap = np.load(
            Path('datasets', ds + '-depth', mode, filename + '.npy'))

        if flip:
            # flip the depthmap as the image is also flipped
            depthmap = np.fliplr(depthmap)

        boxes = scale_boxes(boxes, image.shape[-2:])

        display = normalize(image.clone()*0.5+0.5, (0, 255)).type(torch.uint8)
        display = draw_bounding_boxes(display, boxes)

        plt.imshow(display.permute(1, 2, 0))
        plt.show()
