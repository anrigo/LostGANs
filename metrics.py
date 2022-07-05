import argparse
import os
import pathlib
import numpy as np
import torch
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.inception import InceptionScore
import torchvision.transforms as TF
from tqdm import tqdm
from data.datasets import get_image_files, ImagePathDataset


def get_loader(path, batch_size, num_workers):
    files = get_image_files(path)

    dataset = ImagePathDataset(files, transforms=TF.Resize(
        (299, 299)))  # resize to 299x299 as in original paper
    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=batch_size,
                                             shuffle=False,
                                             drop_last=False,
                                             num_workers=num_workers)

    return dataloader


def main(args):
    device = torch.device('cuda' if (torch.cuda.is_available()) else 'cpu')

    batch_size = 50
    num_workers = os.cpu_count()

    fid = FrechetInceptionDistance(feature=2048).to(device)
    inception = InceptionScore().to(device)

    # load images
    dataloader_real = get_loader(args.real_path, batch_size, num_workers)
    dataloader_fake = get_loader(args.fake_path, batch_size, num_workers)

    for batch in tqdm(dataloader_real):
        batch = batch.to(device)

        fid.update(batch, real=True)

    for batch in tqdm(dataloader_fake):
        batch = batch.to(device)

        fid.update(batch, real=False)
        inception.update(batch)

    # FID
    print(f'FID: {fid.compute()}')

    # IS
    print(f'IS: {inception.compute()[0]}')

    #                               FID     IS
    # clevr-occs-baseline           84.83   2.41
    # clevr-occs-depth-latent       81.29   2.38


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--real_path', type=str, default='',
                        help='path to real images')
    parser.add_argument('--fake_path', type=str, default='',
                        help='path to fake images')
    args = parser.parse_args()
    main(args)
