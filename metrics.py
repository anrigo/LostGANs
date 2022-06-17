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


device = torch.device('cuda' if (torch.cuda.is_available()) else 'cpu')

path_real = 'datasets/coco/images/val2017/'
# path_fake = 'samples/coco128-65/'
# path_fake = 'samples/coco128-30/'
path_fake = 'samples/coco-depth128-55/'
# path_fake = 'samples/coco128-55/'

batch_size = 50
num_workers = 10

fid = FrechetInceptionDistance(feature=2048).to(device)
inception = InceptionScore().to(device)


# real images
dataloader = get_loader(path_real, batch_size, num_workers)

for batch in tqdm(dataloader):
    batch = batch.to(device)

    fid.update(batch, real=True)


# fake images
dataloader = get_loader(path_fake, batch_size, num_workers)

for batch in tqdm(dataloader):
    batch = batch.to(device)

    fid.update(batch, real=False)
    inception.update(batch)

# FID
# 90.6017 COCO 128 30 eps
# 86.6684 COCO 128 55 eps
# 85.2674 COCO 128 55 eps, depth latent
# 83.2490 COCO 128 65 eps

print(f'FID: {fid.compute()}')


# IS
# 12.2888 COCO 128 30 eps
# 12.8534 COCO 128 55 eps
# 12.7928 COCO 128 55 eps, depth latent
# 13.2863 COCO 128 65 eps

print(f'IS: {inception.compute()[0]}')
