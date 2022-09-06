import argparse
from collections import OrderedDict
from operator import index
import os
from pathlib import Path
import shutil
from imageio import imsave
import torch
from tqdm import tqdm
from model.resnet_generator_v2 import ResnetGenerator128, ResnetGeneratorTransfFeats128
from utils.evaluation import compute_metrics
from data.datasets import get_dataset, get_num_classes_and_objects
from utils.util import truncted_random
from pandas import DataFrame


def sample_test(netG, dataset, num_obj, sample_path, lpips_samples=100):
    '''Samples images from the model using the provided split layouts and saves them in sample_path'''
    netG.eval()

    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=1,
        drop_last=True, shuffle=False, num_workers=1)

    alt_path = Path(sample_path, 'alt')

    if not alt_path.is_dir():
        os.makedirs(alt_path)
    thres = 2.0

    for idx, data in tqdm(enumerate(dataloader), total=len(dataset), desc=f'Sampling {len(dataset)} fake images in {sample_path}'):

        if dataset.return_depth:
            real_images, label, bbox, depths = data
            depths = depths.cuda()
        else:
            real_images, label, bbox = data

        real_images, label = real_images.cuda(), label.long().unsqueeze(-1).cuda()
        z_obj = torch.from_numpy(truncted_random(
            num_o=num_obj, thres=thres)).float().cuda()
        z_im = torch.from_numpy(truncted_random(
            num_o=1, thres=thres)).view(1, -1).float().cuda()

        z_obj_alt = torch.from_numpy(truncted_random(
            num_o=num_obj, thres=thres)).float().cuda()
        z_im_alt = torch.from_numpy(truncted_random(
            num_o=1, thres=thres)).view(1, -1).float().cuda()

        if dataset.return_depth:
            fake_images = netG.forward(
                z_obj, bbox.cuda(), z_im=z_im, y=label.squeeze(dim=-1), depths=depths)

            if lpips_samples >= 0:
                # generate different image from the same layout
                fake_images_alt = netG.forward(
                    z_obj_alt, bbox.cuda(), z_im=z_im_alt, y=label.squeeze(dim=-1), depths=depths)
                lpips_samples -= 1
        else:
            fake_images = netG.forward(
                z_obj, bbox.cuda(), z_im, label.squeeze(dim=-1))

            if lpips_samples >= 0:
                # generate different image from the same layout
                fake_images_alt = netG.forward(
                    z_obj_alt, bbox.cuda(), z_im_alt, label.squeeze(dim=-1))
                lpips_samples -= 1

        # normalize from [-1,1] to [0,255]
        result = ((fake_images[0].detach().permute(
            1, 2, 0) + 1) / 2 * 255).type(torch.uint8).cpu().numpy()

        imsave(
            "{save_path}/sample_{idx}.jpg".format(save_path=sample_path, idx=idx), result)

        if lpips_samples >= 0:
            # save the alternative image

            # normalize from [-1,1] to [0,255]
            result_alt = ((fake_images_alt[0].detach().permute(
                1, 2, 0) + 1) / 2 * 255).type(torch.uint8).cpu().numpy()

            imsave(
                "{save_path}/sample_{idx}.jpg".format(save_path=alt_path, idx=idx), result_alt)


def model_sel(args):
    '''Computes validation fid on multiple checkpoints of a model, then computes test fid on the best model'''
    weights_path = os.path.join(
        'outputs', args.dataset + '-' + args.model, 'model')
    models = os.listdir(weights_path)
    results = {'model': [], 'val_fid': []}

    for ii, model in enumerate(models):
        print(f'[{ii+1}/{len(models)}] Evaluating {args.model} checkpoints')
        args.model_path = os.path.join(weights_path, model)
        print(f'Evaluating {args.model_path}')
        metrics = test(args, 'val')
        results['model'].append(model)
        results['val_fid'].append(metrics['val_fid'])

    min_idx = results['val_fid'].index(min(results['val_fid']))

    print(f"Testing {results['model'][min_idx]} on the test set")
    args.model_path = os.path.join(weights_path, results['model'][min_idx])
    metrics = test(args, 'test')
    best_res = {'model': [results['model'][min_idx]],
                'fid (test set)': [metrics['val_fid']]}

    results['model'][min_idx] = f"**{results['model'][min_idx]}**"
    results['val_fid'][min_idx] = f"**{results['val_fid'][min_idx]}**"

    str_res = DataFrame.from_dict(results).to_markdown(index=False)
    str_best = DataFrame.from_dict(best_res).to_markdown(index=False)

    savepath = os.path.join('outputs', args.dataset +
                            '-' + args.model, 'results.txt')
    print(f'Saving to {savepath}')
    with open(savepath, 'w') as f:
        print(f'\nResults for: {args.model}')
        print(str_res)
        f.write(f'Results for: {args.model}\n')
        f.write(str_res)

        print('\nBest model')
        print(str_best)
        f.write('\n\nBest model\n')
        f.write(str_best)


def test(args, split='test'):
    num_classes, num_obj = get_num_classes_and_objects(args.dataset)

    # output directory samples/dataset-model_name
    args.sample_path = os.path.join(
        args.sample_path, args.dataset + '-' + args.model)

    # get test dataset
    dataset = get_dataset(args.dataset, 128, split,
                          num_obj=num_obj,
                          return_depth=args.use_depth)

    # load model
    if args.use_depth:
        netG = ResnetGeneratorTransfFeats128(
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
    metrics_dict = compute_metrics(
        dataset.image_dir, args.sample_path, 50)

    print(f'METRICS: {metrics_dict}')

    if not args.keep:
        # clean
        print('Cleaning')
        shutil.rmtree(args.sample_path)

    return metrics_dict


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
    parser.add_argument('--keep', action=argparse.BooleanOptionalAction,
                        default=False, help='if true, the sampled images won\'t be deleted')
    parser.add_argument('--model', type=str, default='baseline',
                        help='short model name')
    parser.add_argument('--sel', action=argparse.BooleanOptionalAction,
                        default=False, help='if true, runs model selection')
    args = parser.parse_args()

    # args.dataset = 'clevr-rubber'
    # args.model_path = 'outputs/clevr-rubber-baseline-color-lbl/G_200.pth'
    # args.use_depth = True
    # args.model = 'depth-latent'
    # args.keep = True

    if args.sel:
        model_sel(args)
    else:
        test(args)
