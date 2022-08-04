import numpy as np
import torch
import os
import torch_fidelity
import sklearn.metrics
import cleanfid.fid as clean_fid
from tqdm import tqdm
from pathlib import Path
from data.datasets import ImagePathDataset, get_image_files_in_path, DirDataset
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
import wandb
from torch.utils.data import DataLoader
from sklearn.neighbors import NearestNeighbors
from torchvision.utils import make_grid
from torchvision.transforms import Resize
from cleanfid.inception_torchscript import InceptionV3W


def get_image_path_loader(path, batch_size, num_workers):
    files = get_image_files_in_path(path)

    dataset = ImagePathDataset(files)
    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=batch_size,
                                             shuffle=False,
                                             drop_last=False,
                                             num_workers=num_workers)

    return dataloader


def compute_metrics(real_path, fake_path, batch_size, num_workers=24, device=torch.device('cuda')):
    '''Given two paths to real and fake images, computes metrics on them'''

    # get fake and alt files in the same order
    l_alt = get_image_files_in_path(Path(fake_path, 'alt'))
    l_fake = [Path(p.parents[1], p.name) for p in l_alt]

    # real files
    l_real = get_image_files_in_path(real_path)

    fake_ds = ImagePathDataset(l_fake)
    alt_ds = ImagePathDataset(l_alt)
    real_ds = ImagePathDataset(l_real)

    # if fake images are smaller than InceptionV3 input size 299
    # and real images are bigger
    rh, rw = real_ds[0].shape[-2:]
    fake_size = fake_ds[0].shape[-1]
    to_size = None

    if (rh > 299 or rw > 299) and fake_size < 299:
        to_size = fake_size
        real_ds = ImagePathDataset(l_real, to_uint8=True, size=to_size)

    print('Computing FID and IS')
    isc_fid_dict = torch_fidelity.calculate_metrics(
        input1=fake_path,
        input2=real_ds if to_size is not None else real_path,
        cuda=(device.type == 'cuda'),
        isc=True,
        fid=True
    )

    # rename results for easier comparison with previous experiments
    isc_fid_dict.pop('inception_score_std')
    isc_fid_dict['val_fid'] = isc_fid_dict.pop('frechet_inception_distance')
    isc_fid_dict['val_is'] = isc_fid_dict.pop('inception_score_mean')

    print('Computing LPIPS')
    lpips = LearnedPerceptualImagePatchSimilarity(net_type='vgg').to(device)
    
    # compute lpips on pairs of images generated from the same layout
    for imgs in tqdm(zip(fake_ds, alt_ds)):
        fake, alt = imgs

        # normalize from [0,255] to [-1,1] as required by the metric
        fake, alt = ((fake.to(device)/255*2)-1).unsqueeze(0), ((alt.to(device)/255*2)-1).unsqueeze(0)
        lpips.update(fake, alt)

    lpips_res = lpips.compute().item()

    if to_size is not None:
        resizer = clean_fid.make_resizer("PIL", False, "bilinear", (to_size,to_size))
        def resize_(x):
            h, w = x.shape[:2]
            if h > 299 or w > 299:
                return resizer(x)
            return x

    print('Computing Clean FID')
    # compute clean FID
    # the Resize operation (custom image transform) it's applied before anything else
    cfid, np_real_feats, np_fake_feats = cleanfid_compute_fid_return_feat(
        real_path, fake_path, batch_size=batch_size, device=device, num_workers=num_workers,
        custom_image_tranform=resize_ if to_size is not None else None
        )

    print('Computing Precision, Recall, Density and Coverage')
    # compute k-NN based precision precision, recall, density, and coverage
    prdc = compute_prdc(np_real_feats, np_fake_feats, 10)

    # merge all results in a single dictionary and return
    return {**isc_fid_dict, 'lpips': lpips_res, 'clean_fid': cfid, **prdc}


"""
returns a function that takes an image in range [0,255]
and outputs a feature embedding vector
"""
def build_clean_fid_feature_extractor(name="torchscript_inception", device=torch.device("cuda"), resize_inside=False):
    path = 'outputs'
    model = InceptionV3W(path, download=True, resize_inside=resize_inside).to(device)
    model.eval()
    def model_fn(x): return model(x)
    return model_fn


def cleanfid_compute_fid_return_feat(fdir1, fdir2, mode='clean', num_workers=0,
                                     batch_size=8, device=torch.device("cuda"), verbose=True,
                                     custom_image_tranform=None):

    feat_model = build_clean_fid_feature_extractor(mode, device)

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


# PRDC

# modified from https://github.com/facebookresearch/ic_gan/blob/8eff2f7390e385e801993211a941f3592c50984d/data_utils/compute_pdrc.py#L1

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
    distance_real_fake = compute_pairwise_distance(
        real_features, fake_features)

    precision = (
        (distance_real_fake < np.expand_dims(
            real_nearest_neighbour_distances, axis=1))
        .any(axis=0)
        .mean()
    )

    recall = (
        (distance_real_fake < np.expand_dims(
            fake_nearest_neighbour_distances, axis=0))
        .any(axis=1)
        .mean()
    )

    density = (1.0 / float(nearest_k)) * (
        distance_real_fake < np.expand_dims(
            real_nearest_neighbour_distances, axis=1)
    ).sum(axis=0).mean()

    coverage = (
        distance_real_fake.min(axis=1) < real_nearest_neighbour_distances
    ).mean()

    return dict(precision=precision, recall=recall, density=density, coverage=coverage, prdc_image_num=real_features.shape[0])


# KNN VIS

def knn_vis(real_path, fake_path, dataset_name, size, batch_size=10):
    fid_kwargs = dict()
    fid_kwargs['dataset_name'] = dataset_name
    fid_kwargs['image_size'] = size

    return get_knn_eval_dict(sample_dir=fake_path, gt_dir_4fid=real_path,
                             fid_kwargs=fid_kwargs, batch_size=batch_size)


def get_feat_extract_fn(fid_kwargs, name='simclr'):

    if name == 'inception':
        device = torch.device('cuda')
        feat_model = clean_fid.build_feature_extractor('clean', device)
        resize_ = Resize(299)

        def feat_extract_fn(_imgs):
            _feats = clean_fid.get_batch_features(
                resize_(_imgs), feat_model, device)
            return torch.from_numpy(_feats)

        return feat_extract_fn
    else:
        raise

    # if name=='simclr':
    #     feat_backbone = simclr_4sg(dataset_name=fid_kwargs['dataset_name'], image_size=fid_kwargs['image_size'])
    #     def feat_extract_fn(_imgs):
    #         batch_transformed = feat_backbone.transform_batch(_imgs)
    #         _feats = feat_backbone.batch_encode_feat(batch_transformed)['feat'].cpu()
    #         return _feats
    #     return feat_extract_fn
    # elif name=='image':
    #     feat_backbone = simclr_4sg(dataset_name=fid_kwargs['dataset_name'], image_size=fid_kwargs['image_size'])
    #     def feat_extract_fn(_imgs):
    #         raise
    #         return _imgs
    #     return feat_extract_fn
    # else:raise


def get_feat_and_imgs(feat_extract_fn, image_dir, batch_size):
    ds = DataLoader(DirDataset(image_dir), batch_size=batch_size)
    feats, imgs = [], []
    for batch_id, _imgs in tqdm(enumerate(ds), desc='extracting info for KNN vis'):
        with torch.no_grad():
            _feats = feat_extract_fn(_imgs)
            feats.append(_feats)
            imgs.append(_imgs.cpu())
    feats, imgs = torch.cat(feats, 0), torch.cat(imgs, 0)  # [B, C]
    feats = torch.nn.functional.normalize(
        feats, dim=1, p=2.0)  # Normalize feat
    return feats, imgs


def get_wandbimg_by_knn(feats_q, imgs_q, imgs_gallery, nn_obj_gallery):
    query_wandb_result = []
    query_result = nn_obj_gallery.kneighbors(feats_q, return_distance=False)
    for _i in range(query_result.shape[0]):
        result_idx = query_result[_i]
        # indexing result and removing itself
        result_img_row = imgs_gallery[result_idx, :]
        cur_row = np.concatenate(
            [imgs_q[_i:_i + 1], result_img_row], 0)  # [B,3,W,H]
        query_row_concated = make_grid(torch.tensor(cur_row, dtype=torch.uint8), nrow=cur_row.shape[0],
                                       scale_each=True, pad_value=255)
        query_wandb_result.append(wandb.Image(query_row_concated / 255.0))
    return query_wandb_result


def get_knn_eval_dict(sample_dir, gt_dir_4fid, fid_kwargs, knn_k=16, vis_knn=10, batch_size=1, debug=False):
    # KNN
    if debug:
        knn_k = 4
    nn_obj_sample = NearestNeighbors(n_neighbors=knn_k)
    nn_obj_gt = NearestNeighbors(n_neighbors=knn_k)
    nn_obj_all = NearestNeighbors(n_neighbors=knn_k)
    print(f'running KNN for sample dir: {sample_dir}, gt_dir: {gt_dir_4fid}')
    knn_dict = dict()
    for feat_name in ['inception']:
        feat_extract_fn = get_feat_extract_fn(
            fid_kwargs=fid_kwargs, name=feat_name)
        assert batch_size > 1
        feats_sample, imgs_sample = get_feat_and_imgs(
            feat_extract_fn=feat_extract_fn, image_dir=sample_dir, batch_size=batch_size)
        feats_gt, imgs_gt = get_feat_and_imgs(
            feat_extract_fn=feat_extract_fn, image_dir=gt_dir_4fid, batch_size=batch_size)
        # set desired number of neighbors
        nn_obj_sample.fit(feats_sample.cpu())
        nn_obj_gt.fit(feats_gt.cpu())
        if False:
            feats_all = torch.cat([feats_sample, feats_gt], 0)
            imgs_all = torch.cat([imgs_sample, imgs_gt], 0)
            nn_obj_all.fit(feats_all.cpu())

        query_wandb_result = get_wandbimg_by_knn(
            feats_q=feats_sample[:vis_knn], imgs_q=imgs_sample[:vis_knn], imgs_gallery=imgs_gt, nn_obj_gallery=nn_obj_gt)
        knn_dict[f'knn_{feat_name}_query_gt_by_sample'] = query_wandb_result
        query_wandb_result = get_wandbimg_by_knn(
            feats_q=feats_gt[:vis_knn], imgs_q=imgs_gt[:vis_knn], imgs_gallery=imgs_sample, nn_obj_gallery=nn_obj_sample)
        knn_dict[f'knn_{feat_name}_query_sample_by_gt'] = query_wandb_result

    return knn_dict
    # KNN
