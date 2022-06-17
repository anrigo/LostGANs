import argparse
from collections import OrderedDict
from matplotlib import pyplot as plt
import numpy as np
# from scipy import misc
from imageio import imsave
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from data.cocostuff_loader import *
from data.vg import *
from model.resnet_generator_v2 import *
from utils.util import *
from data.datasets import get_dataset


def get_dataloader(dataset = 'coco', img_size=128, return_depth=False):
    
    dataset = get_dataset(dataset, img_size, 'val', return_depth=return_depth)

    dataloader = torch.utils.data.DataLoader(
                    dataset, batch_size=1,
                    drop_last=True, shuffle=False, num_workers=1)
    return dataloader


def main(args):
    num_classes = 184 if args.dataset == 'coco' else 179
    num_o = 8 if args.dataset == 'coco' else 31

    dataloader = get_dataloader(args.dataset, return_depth=args.use_depth)

    if args.use_depth:
        netG = ResnetGeneratorDepth128(num_classes=num_classes, output_dim=3).cuda()
    else:
        netG = ResnetGenerator128(num_classes=num_classes, output_dim=3).cuda()

    if not os.path.isfile(args.model_path):
        print('Model not found')
        raise FileNotFoundError('Model not found')

    state_dict = torch.load(args.model_path)

    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:]  # remove `module.`nvidia
        new_state_dict[name] = v

    model_dict = netG.state_dict()
    pretrained_dict = {k: v for k, v in new_state_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    netG.load_state_dict(model_dict)

    netG.cuda()
    netG.eval()

    if not os.path.exists(args.sample_path):
        os.makedirs(args.sample_path)
    thres=2.0

    for idx, data in tqdm(enumerate(dataloader)):

        if args.use_depth:
            real_images, label, bbox, depths = data
            depths = depths.cuda()
        else:
            real_images, label, bbox = data

        real_images, label = real_images.cuda(), label.long().unsqueeze(-1).cuda()
        z_obj = torch.from_numpy(truncted_random(num_o=num_o, thres=thres)).float().cuda()
        z_im = torch.from_numpy(truncted_random(num_o=1, thres=thres)).view(1, -1).float().cuda()

        if args.use_depth:
            fake_images = netG.forward(z_obj, bbox.cuda(), z_im=z_im, y=label.squeeze(dim=-1), depths=depths)
        else:
            fake_images = netG.forward(z_obj, bbox.cuda(), z_im, label.squeeze(dim=-1))
        
        result = normalize_tensor(fake_images[0].detach().permute(1, 2, 0)*0.5+0.5, (0,255)).type(torch.uint8).cpu().numpy()

        imsave("{save_path}/sample_{idx}.jpg".format(save_path=args.sample_path, idx=idx), result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='coco',
                        help='training dataset')
    parser.add_argument('--model_path', type=str,
                        help='which epoch to load')
    parser.add_argument('--sample_path', type=str, default='samples',
                        help='path to save generated images')
    parser.add_argument('--use_depth', type=bool, default=False,
                        help='use depth')
    args = parser.parse_args()
    main(args)
