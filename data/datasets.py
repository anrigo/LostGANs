import json
from pathlib import Path
from data.cocostuff_loader import CocoSceneGraphDataset
from data.vg import VgSceneGraphDataset


def get_dataset(dataset, img_size, mode=None, depth_dir=None, num_obj=None, return_filenames=False, return_depth=False):

    num_classes = 184 if dataset == 'coco' else 179

    if depth_dir is None:
        depth_dir = Path('datasets', dataset + '-depth', mode)

    if num_obj is None:
        num_obj = 8 if dataset == 'coco' else 31

    if mode is None or mode == 'train':
        coco_image_dir = './datasets/coco/images/train2017/'
        coco_instances_json = './datasets/coco/annotations/instances_train2017.json'
        coco_stuff_json = './datasets/coco/annotations/stuff_train2017.json'
        vg_h5_path = './datasets/vg/train.h5'
        vg_image_dir = './datasets/vg/images/'
    elif mode == 'val':
        coco_image_dir = './datasets/coco/images/val2017/'
        coco_instances_json = './datasets/coco/annotations/instances_val2017.json'
        coco_stuff_json = './datasets/coco/annotations/stuff_val2017.json'
        vg_h5_path = './datasets/vg/val.h5'
        vg_image_dir = './datasets/vg/images/'

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

        data = VgSceneGraphDataset(vocab=vocab, h5_path=vg_h5_path,
                                   image_dir=vg_image_dir,
                                   image_size=(img_size, img_size), max_objects=num_obj-1, left_right_flip=True)
    return data
