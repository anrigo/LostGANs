import argparse
from collections import OrderedDict
import shutil
# from scipy import misc
from imageio import imsave
import torch
from tqdm import tqdm
from data.cocostuff_loader import *
from data.vg import *
from model.resnet_generator_v2 import *
from utils.util import *
import torchvision.transforms as TF
from data.datasets import ImagePathDataset, get_dataset, get_image_files_in_path, get_num_classes_and_objects
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.inception import InceptionScore


def get_image_path_loader(path, batch_size, num_workers):
    files = get_image_files_in_path(path)

    dataset = ImagePathDataset(files, transforms=TF.Resize(
        (299, 299)))  # resize to 299x299 as in original paper
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
        real_path, 50, os.cpu_count())
    fake_dataloader = get_image_path_loader(
        fake_path, 50, os.cpu_count())

    fid = FrechetInceptionDistance(feature=2048).to(device)
    inception = InceptionScore().to(device)

    for batch in tqdm(real_dataloader):
        batch = batch.to(device)

        fid.update(batch, real=True)

    for batch in tqdm(fake_dataloader):
        batch = batch.to(device)

        fid.update(batch, real=False)
        inception.update(batch)

    return fid.compute(), inception.compute()[0]


def sample_test(netG, dataset, num_obj, sample_path):
    '''Samples images from the model using the provided split layouts and saves them in sample_path'''

    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=1,
        drop_last=True, shuffle=False, num_workers=1)

    if not os.path.exists(sample_path):
        os.makedirs(sample_path)
    thres = 2.0

    for idx, data in tqdm(enumerate(dataloader)):

        if args.use_depth:
            real_images, label, bbox, depths = data
            depths = depths.cuda()
        else:
            real_images, label, bbox = data

        real_images, label = real_images.cuda(), label.long().unsqueeze(-1).cuda()
        z_obj = torch.from_numpy(truncted_random(
            num_o=num_obj, thres=thres)).float().cuda()
        z_im = torch.from_numpy(truncted_random(
            num_o=1, thres=thres)).view(1, -1).float().cuda()

        if args.use_depth:
            fake_images = netG.forward(
                z_obj, bbox.cuda(), z_im=z_im, y=label.squeeze(dim=-1), depths=depths)
        else:
            fake_images = netG.forward(
                z_obj, bbox.cuda(), z_im, label.squeeze(dim=-1))

        # normalize from [-1,1] to [0,255]
        result = ((fake_images[0].detach().permute(
            1, 2, 0) + 1) / 2 * 255).type(torch.uint8).cpu().numpy()

        imsave(
            "{save_path}/sample_{idx}.jpg".format(save_path=args.sample_path, idx=idx), result)


def main(args):
    num_classes, num_obj = get_num_classes_and_objects(args.dataset)

    # output directory samples/dataset-model_name
    args.sample_path = os.path.join(
        args.sample_path, args.dataset + '-' + args.model)

    # get test dataset
    dataset = get_dataset(args.dataset, None, 'test',
                          return_depth=args.use_depth)

    # load model
    if args.use_depth:
        netG = ResnetGeneratorDepth128(
            num_classes=num_classes, output_dim=3).cuda()
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
    pretrained_dict = {k: v for k,
                       v in new_state_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    netG.load_state_dict(model_dict)

    netG.cuda()
    netG.eval()

    # Sample fake images
    print(f'Sampling {len(dataset)} fake images')
    sample_test(netG, dataset, num_obj, args.sample_path)

    # compute metrics
    print('Computing metrics')
    fid, is_ = compute_metrics(dataset.image_dir, args.sample_path, 50, os.cpu_count())

    print(f'FID: {fid}, IS: {is_}')

    # clean
    print('Cleaning')
    shutil.rmtree(args.sample_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='coco',
                        help='training dataset')
    parser.add_argument('--model_path', type=str,
                        help='which epoch to load')
    parser.add_argument('--sample_path', type=str, default='samples',
                        help='path to save generated images')
    parser.add_argument('--use_depth', action=argparse.BooleanOptionalAction,
                        default=False, help='use depth information')
    parser.add_argument('--model', type=str, default='baseline',
                        help='short model name')
    args = parser.parse_args()
    main(args)
