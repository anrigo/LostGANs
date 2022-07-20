import numpy as np
import torch
import os
import sklearn.metrics
import cleanfid.fid as clean_fid
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

    print('Computing LPIPS')
    # compute lpips on pairs of images generated from the same layout
    for imgs in tqdm(zip(fake_ds, alt_ds)):
        fake, alt = imgs

        # normalize from [0,255] to [-1,1] as required by the metric
        fake, alt = ((fake/255*2)-1).unsqueeze(0), ((alt/255*2)-1).unsqueeze(0)
        lpips.update(fake, alt)

    print('Computing FID and IS')
    # iterate over real images and update statistics for FID
    for batch in tqdm(real_dataloader):
        batch = batch.to(device)

        fid.update(batch, real=True)

    # iterate over fake images and update statistics for FID and IS
    for batch in tqdm(fake_dataloader):
        batch = batch.to(device)

        fid.update(batch, real=False)
        inception.update(batch)

    print('Computing Clean FID and PRDC')
    # compute clean FID
    cfid, np_real_feats, np_fake_feats = cleanfid_compute_fid_return_feat(real_path, fake_path)

    # compute k-NN based precision precision, recall, density, and coverage
    prdc = compute_prdc(np_real_feats, np_fake_feats)

    return fid.compute(), inception.compute()[0], lpips.compute(), cfid, prdc


def cleanfid_compute_fid_return_feat(fdir1, fdir2, mode='clean', num_workers=0,
                    batch_size=8, device=torch.device("cuda"), verbose=True,
                    custom_image_tranform=None):

    feat_model = clean_fid.build_feature_extractor(mode, device)
    
    # get all inception features for the first folder
    fbname1 = os.path.basename(fdir1)
    np_feats1 = clean_fid.get_folder_features(fdir1, feat_model, num_workers=num_workers,
                                    batch_size=batch_size, device=device, mode=mode,
                                    description=f"FID {fbname1} : ", verbose=verbose,
                                    custom_image_tranform=custom_image_tranform)
    mu1 = np.mean(np_feats1, axis=0)
    sigma1 = np.cov(np_feats1, rowvar=False)
    # get all inception features for the second folder
    fbname2 = os.path.basename(fdir2)
    np_feats2 = clean_fid.get_folder_features(fdir2, feat_model, num_workers=num_workers,
                                    batch_size=batch_size, device=device, mode=mode,
                                    description=f"FID {fbname2} : ", verbose=verbose,
                                    custom_image_tranform=custom_image_tranform)
    mu2 = np.mean(np_feats2, axis=0)
    sigma2 = np.cov(np_feats2, rowvar=False)
    fid = clean_fid.frechet_distance(mu1, sigma1, mu2, sigma2)
    return fid, np_feats1, np_feats2


#modified from https://github.com/facebookresearch/ic_gan/blob/8eff2f7390e385e801993211a941f3592c50984d/data_utils/compute_pdrc.py#L1

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# prdc
# Copyright (c) 2020-present NAVER Corp.
# MIT license


def compute_pairwise_distance(data_x, data_y=None):
    """
    Parameters
    ----------
        data_x: numpy.ndarray([N, feature_dim], dtype=np.float32)
        data_y: numpy.ndarray([N, feature_dim], dtype=np.float32)
    Returns
    -------
        numpy.ndarray([N, N], dtype=np.float32) of pairwise distances.
    """
    if data_y is None:
        data_y = data_x
    dists = sklearn.metrics.pairwise_distances(
        data_x, data_y, metric="euclidean", n_jobs=8
    )
    return dists


def get_kth_value(unsorted, k, axis=-1):
    """
    Parameters
    ----------
        unsorted: numpy.ndarray of any dimensionality.
        k: int
        axis: int
    Returns
    -------
        kth values along the designated axis.
    """
    indices = np.argpartition(unsorted, k, axis=axis)[..., :k]
    k_smallests = np.take_along_axis(unsorted, indices, axis=axis)
    kth_values = k_smallests.max(axis=axis)
    return kth_values


def compute_nearest_neighbour_distances(input_features, nearest_k):
    """
    Parameters
    ----------
        input_features: numpy.ndarray([N, feature_dim], dtype=np.float32)
        nearest_k: int
    Returns
    -------
        Distances to kth nearest neighbours.
    """
    distances = compute_pairwise_distance(input_features)
    radii = get_kth_value(distances, k=nearest_k + 1, axis=-1)
    return radii


def compute_prdc(real_features, fake_features, nearest_k=5):
    """
    Computes precision, recall, density, and coverage given two manifolds.

    Parameters
    ----------
        real_features: numpy.ndarray([N, feature_dim], dtype=np.float32)
        fake_features: numpy.ndarray([N, feature_dim], dtype=np.float32)
        nearest_k: int.
    Returns
    -------
        dict of precision, recall, density, and coverage.
    """

    print(
        "Num real: {} Num fake: {}".format(
            real_features.shape[0], fake_features.shape[0]
        )
    )

    real_nearest_neighbour_distances = compute_nearest_neighbour_distances(
        real_features, nearest_k
    )
    fake_nearest_neighbour_distances = compute_nearest_neighbour_distances(
        fake_features, nearest_k
    )
    distance_real_fake = compute_pairwise_distance(real_features, fake_features)

    precision = (
        (distance_real_fake < np.expand_dims(real_nearest_neighbour_distances, axis=1))
        .any(axis=0)
        .mean()
    )

    recall = (
        (distance_real_fake < np.expand_dims(fake_nearest_neighbour_distances, axis=0))
        .any(axis=1)
        .mean()
    )

    density = (1.0 / float(nearest_k)) * (
        distance_real_fake < np.expand_dims(real_nearest_neighbour_distances, axis=1)
    ).sum(axis=0).mean()

    coverage = (
        distance_real_fake.min(axis=1) < real_nearest_neighbour_distances
    ).mean()

    return dict(precision=precision, recall=recall, density=density, coverage=coverage, prdc_image_num=real_features.shape[0])

