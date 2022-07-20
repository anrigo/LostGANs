import torch
import os
from tqdm import tqdm
from pathlib import Path
from data.datasets import ImagePathDataset, get_image_files_in_path
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.inception import InceptionScore
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity


def get_image_path_loader(path, batch_size, num_workers):
    files = get_image_files_in_path(path)

    dataset = ImagePathDataset(files)
    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=batch_size,
                                             shuffle=False,
                                             drop_last=False,
                                             num_workers=num_workers)

    return dataloader


def compute_metrics(real_path, fake_path, batch_size, num_workers, device='cuda'):
    '''Given two paths to real and fake images, computes metrics on them'''

    # dataloaders for images only, so they are both in the same format
    real_dataloader = get_image_path_loader(
        real_path, batch_size, num_workers)
    fake_dataloader = get_image_path_loader(
        fake_path, batch_size, num_workers)

    # get files in the same order
    l_alt = get_image_files_in_path(Path(fake_path, 'alt'))
    l_fake = [Path(p.parents[1], p.name) for p in l_alt]

    fake_ds = ImagePathDataset(l_fake)
    alt_ds = ImagePathDataset(l_alt)

    fid = FrechetInceptionDistance(feature=2048).to(device)
    inception = InceptionScore().to(device)
    lpips = LearnedPerceptualImagePatchSimilarity(net_type='vgg')

    # compute lpips on pairs of images generated from the same layout
    for imgs in tqdm(zip(fake_ds, alt_ds)):
        fake, alt = imgs

        # normalize from [0,255] to [-1,1] as required by the metric
        fake, alt = ((fake/255*2)-1).unsqueeze(0), ((alt/255*2)-1).unsqueeze(0)
        lpips.update(fake, alt)

    for batch in tqdm(real_dataloader):
        batch = batch.to(device)

        fid.update(batch, real=True)

    for batch in tqdm(fake_dataloader):
        batch = batch.to(device)

        fid.update(batch, real=False)
        inception.update(batch)

    return fid.compute(), inception.compute()[0], lpips.compute()
