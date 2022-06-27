import json
from pathlib import Path
from typing import Union
import torch
from torch.utils.data import Dataset
from utils.util import normalize_tensor
from PIL import Image
import torchvision.transforms as T
from torch.nn.functional import pad


class CLEVRDataset(Dataset):
    def __init__(self, image_dir: Union[str, Path], scenes_json: Union[str, Path], image_size: tuple[int, int], max_objects_per_image=10, return_depth: bool = False) -> None:
        super(Dataset, self).__init__()

        self.image_dir = image_dir
        self.scenes_json = scenes_json
        self.image_size = image_size
        self.return_depth = return_depth
        self.idx2label = generate_label_map()

        # Each scene contains between 3 and 10 random objects
        self.max_objects_per_image = max_objects_per_image

        # compose transforms to resize and convert to tensor
        transforms = []

        # if image size is set to None, load the original image without resizing
        if image_size[0] is not None:
            transforms.append(T.Resize(image_size))

        transforms.append(T.ToTensor())

        self.transforms = T.Compose(transforms)

        # load scenes data
        with open(scenes_json, 'r') as fj:
            self.scenes = json.load(fj)['scenes']

    def __getitem__(self, index):
        scene = self.scenes[index]
        objs = scene['objects']
        image_path = Path(self.image_dir, scene['image_filename'])

        # load image
        image = Image.open(image_path).convert('RGB')
        image = self.transforms(image)

        # clevr images are in [0,1]
        # normalize to [-1,1]
        image = (image * 2) - 1

        # extract bounding boxes from the objects' coordinates
        bboxes, labels = extract_bounding_boxes(scene, self.idx2label)

        # add dummy objects to reach the desired number
        for _ in range(len(labels), self.max_objects_per_image):
            # label 0: dummy object __image__
            labels.append(0)
            bboxes.append((-0.6, -0.6, 0.5, 0.5))

        # convert everything to tensors
        labels = torch.tensor(labels)
        bboxes = torch.tensor(bboxes)

        if self.return_depth:
            # extract, invert and normalize depth values
            depths = [-1*obj['pixel_coords'][2] for obj in objs]
            depths = normalize_tensor(torch.tensor(depths), (0, 1))

            # set dummy objects depth to -0.5
            depths = pad(depths, (0, self.max_objects_per_image-depths.shape[0]), value=-0.5)

            return image, labels, bboxes, depths

        return image, labels, bboxes

    def __len__(self):
        return len(self.scenes)


def generate_label_map():
    sizes = ['large', 'small']
    colors = ['gray', 'red', 'blue', 'green',
              'brown', 'purple', 'cyan', 'yellow']
    materials = ['rubber', 'metal']
    shapes = ['cube', 'sphere', 'cylinder']

    names = [s + ' ' + c + ' ' + m + ' ' +
             sh for s in sizes for c in colors for m in materials for sh in shapes]

    # add dummy objects' label
    names.insert(0, '__image__')

    return names


def extract_bounding_boxes(scene, names):
    objs = scene['objects']
    rotation = scene['directions']['right']

    xmin = []
    ymin = []
    # xmax = []
    # ymax = []
    widths = []
    heights = []
    classes = []
    classes_text = []

    for _, obj in enumerate(objs):
        [x, y, z] = obj['pixel_coords']

        [x1, y1, z1] = obj['3d_coords']

        cos_theta, sin_theta, _ = rotation

        x1 = x1 * cos_theta + y1 * sin_theta
        y1 = x1 * -sin_theta + y1 * cos_theta

        height_d = 6.9 * z1 * (15 - y1) / 2.0
        height_u = height_d
        width_l = height_d
        width_r = height_d

        if obj['shape'] == 'cylinder':
            d = 9.4 + y1
            h = 6.4
            s = z1

            height_u *= (s*(h/d + 1)) / ((s*(h/d + 1)) - (s*(h-s)/d))
            height_d = height_u * (h-s+d) / (h + s + d)

            width_l *= 11/(10 + y1)
            width_r = width_l

        if obj['shape'] == 'cube':
            height_u *= 1.3 * 10 / (10 + y1)
            height_d = height_u
            width_l = height_u
            width_r = height_u

        obj_name = obj['size'] + ' ' + obj['color'] + \
            ' ' + obj['material'] + ' ' + obj['shape']
        classes_text.append(obj_name.encode('utf8'))

        classes.append(names.index(obj_name))

        ymin.append((y - height_d)/320.0)
        # ymax.append((y + height_u)/320.0)
        xmin.append((x - width_l)/480.0)
        # xmax.append((x + width_r)/480.0)

        heights.append((height_u + height_d)/320.0)
        widths.append((width_l + width_r)/480.0)

    bboxes = [(xmin[i], ymin[i], widths[i], heights[i])
              for i in range(len(xmin))]

    return bboxes, classes
