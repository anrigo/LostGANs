from operator import itemgetter
import os
from pathlib import Path
import cv2
import numpy as np
import torch
from tqdm import tqdm
from utils.util import normalize_tensor, scale_boxes
from torchvision.transforms.functional import crop
from torch.nn.functional import pad

device = torch.device(
    "cuda") if torch.cuda.is_available() else torch.device("cpu")


def depth_estimation(dataset, ds, mode, save=False, limit=None):
    '''
    Use MiDaS Large to estimate depth from each image in the dataset and save
    the depthmaps as .npy files, currently works for coco

    Args:
        dataset: Dataset object from get_dataset, returning the images' filenames
        ds: name of the dataset, coco or vg
        mode: train or val
        visualize: visualize the resulting depthmaps
        save: save depthmaps
        limit: how many images to process 
    '''

    # # load dataset
    # dataset = get_dataset(ds, None, mode, return_filenames=True)

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

            # save depthmap
            if save:
                np.save(Path(save_path, filename + '.npy'), prediction)


def get_bboxes_depths_from_depthmap(depthmap: torch.Tensor, boxes: torch.Tensor) -> torch.Tensor:
    '''
    Computes depth values for each bounding box from the depthmap
    '''

    # save number of boxes
    num_o = boxes.shape[0]

    # exclude dummy objects
    # always in the last rows of the tensor
    boxes = boxes[boxes[:, 0] >= 0]

    # scale boxes to image size
    # boxes with width and height
    size_boxes = scale_boxes(
        boxes, depthmap.shape[-2:], 'inverse_size', dtype=torch.int)

    # crop the boxes from the depthmap
    crops = [crop(depthmap, *(box.tolist()))
             for box in size_boxes]

    # compute mean depth for each crop
    depths = torch.tensor([crop_.mean() for crop_ in crops])

    # normalize depths
    depths = normalize_tensor(depths, (0, 1))

    # set -0.5 depth to dummy objects
    # since they are in the last rows
    # simply append them
    depths = pad(depths, (0, num_o-depths.shape[0]), value=-0.5)

    return depths


def get_depth_layout(depths: torch.Tensor, size: tuple[int, int], boxes: torch.Tensor) -> torch.Tensor:
    '''
    Puts all bounding boxes depths in depth order in a single tensor to be visualized
    '''

    # compose new depthmap for visualization
    # depths as list of tuples (box index, depth value)
    # excluding dummy objects with negative depth
    boxes_depths = [(i, d) for i, d in enumerate(depths[depths >= 0])]
    # sort tuples by depth value, itemgetter gets the depth (position 1) from each tuple
    boxes_depths = sorted(boxes_depths, key=itemgetter(1))

    # boxes with xmax and ymax
    coord_boxes = scale_boxes(
        boxes, size, 'coordinates', dtype=torch.int)

    # add bboxes depths to an empty depthmap in depth order
    depth_layout = torch.zeros(size, device=depths.device)

    for i, d in boxes_depths:
        # get bounding box coordinates
        x, y, xmax, ymax = coord_boxes[i]

        # create a tensor with every element equal to the bbox depth
        # of the same shape of the bounding box
        patch = d.clone().repeat(depth_layout[..., y:ymax, x:xmax].shape)

        # write the new values
        depth_layout[..., y:ymax, x:xmax] = patch

    return depth_layout


def get_depth_layout_batch(depths_batch: torch.Tensor, size: tuple[int, int], boxes_batch: torch.Tensor) -> torch.Tensor:
    '''
    Returns a tensor with b depth maps, one for each sample in the batch

    Parameters:
        - depths: tensor of size (b, num_objects)
        - size: tuple (H, W)
        - boxes: tensor of size (b, num_objetcs, 4)

    Returns:
        - depthmaps: tensor of size (b, H, W)
    '''

    return torch.stack([get_depth_layout(depths, size, boxes)
             for depths, boxes in zip(depths_batch, boxes_batch)], dim=0)
