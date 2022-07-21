import json
from pathlib import Path
from typing import Union
import os
from torchvision.io import read_image
from torch.utils.data import Dataset
import numpy as np
from data.clevr import CLEVRDataset
from data.cocostuff_loader import CocoSceneGraphDataset
from data.vg import VgSceneGraphDataset
import torch
from PIL import Image

IMAGE_EXTENSIONS = {'bmp', 'jpg', 'jpeg', 'pgm', 'png', 'ppm',
                    'tif', 'tiff', 'webp'}


def get_dataset(dataset: str, img_size: int, mode: str = None, depth_dir: Union[str, Path] = None, num_obj: int = None, return_filenames: bool = False, return_depth: bool = False):

    if depth_dir is None:
        depth_dir = Path('datasets', dataset + '-depth', mode)

    if mode is None:
        mode = 'train'

    coco_image_dir = f'./datasets/coco/images/{mode}2017/'
    coco_instances_json = f'./datasets/coco/annotations/instances_{mode}2017.json'
    coco_stuff_json = f'./datasets/coco/annotations/stuff_{mode}2017.json'

    vg_h5_path = './datasets/vg/{mode}.h5'
    vg_image_dir = './datasets/vg/images/'

    clevr_image_dir = f'./datasets/CLEVR_v1.0/images/{mode}'
    clevr_scenes_json = f'./datasets/CLEVR_v1.0/scenes/CLEVR_{mode}_scenes.json'

    clevr_occs_image_dir = f'./datasets/CLEVR_occlusions/images/{mode}'
    clevr_occs_scenes_json = f'./datasets/CLEVR_occlusions/scenes/CLEVR_{mode}_scenes.json'

    clevr_occs2_image_dir = f'./datasets/CLEVR_occlusions2/images/{mode}'
    clevr_occs2_scenes_json = f'./datasets/CLEVR_occlusions2/scenes/CLEVR_{mode}_scenes.json'

    if depth_dir is None:
        depth_dir = Path('datasets', f'{dataset}-depth', mode)

    if dataset == "coco":
        data = CocoSceneGraphDataset(image_dir=coco_image_dir,
                                     instances_json=coco_instances_json,
                                     stuff_json=coco_stuff_json,
                                     depth_dir=depth_dir,
                                     stuff_only=True, image_size=(img_size, img_size), left_right_flip=True,
                                     return_filenames=return_filenames, return_depth=return_depth)
    elif dataset == 'vg':
        with open('./datasets/vg/vocab.json', 'r') as fj:
            vocab = json.load(fj)

        if num_obj is None:
            num_obj = 31

        data = VgSceneGraphDataset(vocab=vocab, h5_path=vg_h5_path,
                                   image_dir=vg_image_dir,
                                   image_size=(img_size, img_size), max_objects=num_obj-1, left_right_flip=True)
    elif dataset == 'clevr':
        data = CLEVRDataset(image_dir=clevr_image_dir,
                            scenes_json=clevr_scenes_json,
                            image_size=(img_size, img_size), return_depth=return_depth)
    elif dataset == 'clevr-occs':
        data = CLEVRDataset(image_dir=clevr_occs_image_dir,
                            scenes_json=clevr_occs_scenes_json,
                            max_objects_per_image=num_obj,
                            image_size=(img_size, img_size), return_depth=return_depth, occlusions=True)
    elif dataset == 'clevr-occs2':
        data = CLEVRDataset(image_dir=clevr_occs2_image_dir,
                            scenes_json=clevr_occs2_scenes_json,
                            max_objects_per_image=num_obj,
                            image_size=(img_size, img_size), return_depth=return_depth, occlusions=True)

    return data


def get_num_classes_and_objects(dataset: str) -> tuple[int, int]:
    """ Returns (number of classes, maximum number of objects per image) for the desired dataset """
    return {
        'coco': (184, 8),
        'vg': (179, 31),
        'clevr': (97, 10),
        'clevr-occs': (4, 10),
        'clevr-occs2': (4, 20)
    }[dataset]


class ImagePathDataset(Dataset):
    def __init__(self, files, transforms=None):
        self.files = files
        self.transforms = transforms

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        path = self.files[i]
        img = Image.open(path).convert('RGB')

        if self.transforms is not None:
            img = self.transforms(img)

        img = torch.from_numpy(np.array(img))
        img = torch.permute(img, (2, 0, 1))

        return img


class DirDataset(Dataset):
    def __init__(self, img_dir):
        self.img_dir = img_dir
        self.file_names = os.listdir(img_dir)
        self.file_names = [filename for filename in self.file_names if Path(
            self.img_dir, filename).is_file()]

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.file_names[idx])
        image = read_image(img_path)
        return image


def get_image_files_in_path(path: Union[str, Path]) -> list[Path]:
    '''Returns a list of paths to all image files in the specified path'''

    path = Path(path)
    files = sorted([file for ext in IMAGE_EXTENSIONS
                    for file in path.glob('*.{}'.format(ext))])

    return files
