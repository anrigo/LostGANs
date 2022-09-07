import math
from pathlib import Path
import torch
import cv2
from data.vg import *
from utils.util import *
import matplotlib.pyplot as plt
from torchvision.utils import draw_bounding_boxes, make_grid
import torchvision.transforms.functional as T
from skimage.transform import pyramid_expand
from PIL import ImageDraw
from typing import Union, Callable
from utils.depth import get_depth_layout
from numpy.random import randint
from tqdm import tqdm
import utils.evaluation as ev
from test import sample_test


# MODULE VARIABLES
args = dataset = dataset_base = idx2name = num_o = num_classes = thres = netGbase = netGdepth = None


def init(args, dataset, idx2name, num_o, num_classes, thres, netGbase, netGdepth):
    '''   Initializes the module with all the information needed   '''

    globalvars = globals()
    globalvars['args'] = args
    globalvars['dataset'] = dataset
    globalvars['idx2name'] = idx2name
    globalvars['num_o'] = num_o
    globalvars['num_classes'] = num_classes
    globalvars['thres'] = thres
    globalvars['netGbase'] = netGbase
    globalvars['netGdepth'] = netGdepth


def draw_bboxes(image: torch.Tensor, bbox: Union[torch.Tensor, np.ndarray], labels: torch.Tensor, text: bool = True, color_transform: list[int] = None, show_idx: bool = False) -> torch.Tensor:
    '''
    Draws bounding boxes on an image, optionally adds labels and objects' indices and colors some specified bounding boxes

    Args:
        image: image
        bbox: bounding boxes coordinates in the format (x, y, w, h)
        labels: labels
        text: True to visualize text labels in each bbox
        show_idx: if True each bbox index (row of the bbox tensor) will be visualized
        color_transform: list of bbox indices representing the bboxes to be colored in red

    Returns:
        image: the image after drawing the bboxes on it
    '''

    if isinstance(bbox, np.ndarray):
        bbox = torch.from_numpy(bbox)

    colors = None
    label_str = None

    # scale boxes to image size
    # boxes with xmax and ymax
    coord_boxes = scale_boxes(
        bbox, image.shape[-2:], 'coordinates', dtype=torch.int)

    if color_transform is not None:
        # all non-transformed bboxes will be white
        colors = [(255, 255, 255) for _ in range(len(labels))]

        # color transformed bboxes in red
        for i in color_transform:
            colors[i] = (255, 0, 0)

    # show object row index in the tensor
    if show_idx:
        label_str = [str(i) for i in range(len(labels))]

    # show textual label
    if text:
        if show_idx:
            # show both text and objects' indices
            label_str = [label_str[i]+'-'+idx2name[labels[i]]
                         for i in range(len(labels))]
        else:
            label_str = [idx2name[l] for l in labels]

    # draw bboxes
    image = draw_bounding_boxes(
        image, coord_boxes, labels=label_str, colors=colors)

    return image


def sample_one(idx: int = None, show_labels: bool = False):
    '''
    Selects a random layout and displays the real image, the fake images generated by both the baseline model and the depth-aware model

    Args:
        idx: image position in the dataset. If none is specified a random one will be selected
        show_labels: if True the labels will be visualized on the layout
    '''

    # if no image is specified, select a random one
    idx = int(np.ceil(np.random.random()*len(dataset)) -
              1) if idx is None else idx
    print(f'Image: {idx}')

    # control plot order and size
    cols, ax_id, figH = (3, 0, 4)
    figsize = (figH*cols, figH)

    _, axs = plt.subplots(1, cols, figsize=figsize)

    real, labels, bbox, depth = dataset[idx]

    # print each object depth value
    print([(i, d.item()) for i, d in enumerate(depth)])

    if isinstance(bbox, np.ndarray):
        bbox = torch.from_numpy(bbox)

    # scale bounding boxes for visualization
    coord_box = scale_boxes(
        bbox, dataset.image_size, 'coordinates', dtype=torch.int)

    # display layout by
    # displaying bounding boxes on a black tensor
    layout = torch.zeros(3, 128, 128).type(torch.uint8)
    layout = draw_bboxes(layout, bbox, labels, show_idx=True, text=show_labels)

    axs[ax_id].imshow(layout.permute(1, 2, 0))
    axs[ax_id].set_title('Layout with indices')
    ax_id += 1

    # get depth layout
    depth_layout = get_depth_layout(depth, real.shape[-2:], bbox)

    axs[ax_id].imshow(depth_layout, cmap='gray')
    axs[ax_id].set_title('Depth layout')
    ax_id += 1

    # normalize from [-1,1] to [0,255]
    real = ((real.cpu() + 1) / 2 * 255).type(torch.uint8)

    # draw boxes
    real = draw_bounding_boxes(real, coord_box)

    axs[ax_id].imshow(real.permute(1, 2, 0))
    axs[ax_id].set_title('Real image with bounding boxes')

    # display fake images
    _, axs = plt.subplots(1, 2, figsize=(figH*2, figH))
    fake_images = []

    # sample noise vectors
    z_obj = torch.from_numpy(truncted_random(
        num_o=num_o, thres=thres)).float().cuda()
    z_im = torch.from_numpy(truncted_random(
        num_o=1, thres=thres)).view(1, -1).float().cuda()

    # sample baseline
    fake = netGbase.forward(
        z_obj, bbox.clone().cuda().unsqueeze(0), z_im, labels.clone().long().cuda())

    # normalize from [-1,1] to [0,1]
    fake_images.append(fake.detach().squeeze().permute(
        1, 2, 0).cpu() * 0.5 + 0.5)

    # sample depth-aware
    fake = netGdepth.forward(z_obj, bbox.clone().cuda().unsqueeze(
        0), z_im=z_im, y=labels.clone().long().cuda(), depths=depth.cuda().unsqueeze(0))

    # normalize from [-1,1] to [0,1]
    fake_images.append(fake.detach().squeeze().permute(
        1, 2, 0).cpu() * 0.5 + 0.5)

    axs[0].imshow(fake_images[0])
    axs[0].set_title('Baseline fake image')
    axs[1].imshow(fake_images[1])
    axs[1].set_title('Depth-aware fake image')

    plt.show()


def direct_comparison(num_gen: int = 8, cols: int = 8, figunitsize: int = 3, visualize_layout: bool = False):
    '''
    Sample `num_gen` layout randomly from the dataset, then generate baseline fakes and depth-aware fakes for each one and display the result in a grid.

    Args:
        num_gen: number of layouts to sample
        cols: number of rows of the final grid
        figunitsize: plot size of a single fake image, before stacking
        visualize_layout: stack the depth layout on top of the two fakes
    '''

    fakes = []

    for idx in randint(0, len(dataset), num_gen):
        real, label, bbox, depth = dataset[idx]
        size = real.shape[-2:]

        if isinstance(bbox, np.ndarray):
            bbox = torch.from_numpy(bbox)

        # sample noise vectors
        z_obj = torch.from_numpy(truncted_random(
            num_o=num_o, thres=thres)).float().cuda()
        z_im = torch.from_numpy(truncted_random(
            num_o=1, thres=thres)).view(1, -1).float().cuda()

        # baseline fake
        fake_images_base = netGbase.forward(z_obj, bbox.clone().cuda().unsqueeze(
            0), z_im, label.long().cuda()).squeeze()

        # depth-aware fake
        fake_images = netGdepth.forward(z_obj, bbox.clone().cuda().unsqueeze(
            0), z_im=z_im, y=label.long().cuda(), depths=depth.clone().cuda().unsqueeze(0)).squeeze()

        # normalize from [-1,1] to [0,255] and convert to PIL image
        real, fake_images_base, fake_images = T.to_pil_image(((real + 1) / 2 * 255).type(torch.uint8).cpu()), T.to_pil_image(
            ((fake_images_base.detach() + 1) / 2 * 255).type(torch.uint8).cpu()), T.to_pil_image(((fake_images.detach() + 1) / 2 * 255).type(torch.uint8).cpu())

        real_draw, base_draw, depth_draw = ImageDraw.Draw(
            real), ImageDraw.Draw(fake_images_base), ImageDraw.Draw(fake_images)

        real_draw.text((0, 0), 'ground truth', (0, 0, 0))
        base_draw.text((0, 0), 'baseline', (0, 0, 0))
        depth_draw.text((0, 0), 'depth-aware', (0, 0, 0))

        # stack tensors: depth-layout, baseline fake, depth-aware fake
        # tensors are now in [0,1]
        if visualize_layout:
            # get depth layout
            depth_layout = get_depth_layout(
                depth, size, bbox).unsqueeze(0)

            fakes.append(torch.cat((
                torch.cat((depth_layout, depth_layout, depth_layout), 0), T.to_tensor(
                    real), T.to_tensor(fake_images_base), T.to_tensor(fake_images)
            ), dim=1))
        else:
            fakes.append(torch.cat((T.to_tensor(real), T.to_tensor(
                fake_images_base), T.to_tensor(fake_images)), dim=1))

    # number of stacked images in each frame
    framestacked = 4 if visualize_layout else 3

    # build a grid and plot
    rows = int(num_gen/cols)
    plt.figure(figsize=(figunitsize*cols, framestacked *
               figunitsize*rows))  # (width, height)
    plt.imshow(make_grid(fakes, nrow=cols).permute(1, 2, 0))
    plt.axis('off')
    plt.show()


def variable_inputs(variable: list[str], num_gen: int = 23, max_cols: int = 6, figsize: tuple[int, int] = (18, 12), layout_idx: int = 0, transform_bbox: list[int] = None, transform: Callable = None, show_idx: bool = False, show_labels: bool = False, use_depth: bool = False):
    '''
    Generates multiple images while keeping all inputs fixed except the ones specified in the `variable` parameter

    Args:
        variable: list of inputs that will vary at each generation. Can be `layout`, `z_obj`, `z_img` or a combination of them
        num_gen: number of images to be generated
        max_cols: max numer of columns used to display the result
        figsize: figure size
        layout_idx: layout index in the dataset
        transform_bbox: list of indices indicating the bboxes to be transformed
        transform: function used to transform the bboxes indicated by `transform_bbox`
        show_idx: if True each bbox index (row of the bbox tensor) will be visualized
        show_labels: labels will be visualized on the layout if set to True
        use_depth: if True depth information will be used as input to the model
    '''

    fakes = []

    # fixed inputs

    if not 'layout' in variable:
        # the layout is a fixed input, take it from the dataset once outside of the loop

        _, label, bbox, depth = dataset[layout_idx]

        if isinstance(bbox, np.ndarray):
            bbox = torch.from_numpy(bbox)

        # display layout by displaying bounding boxes on a black tensor
        layout = torch.zeros(3, 128, 128).type(torch.uint8)

        # if a bbox gets transformed, color it in red on the layout
        if transform_bbox is not None:
            # color transformed bboxes in red
            layout = draw_bboxes(
                layout, bbox, label, color_transform=transform_bbox, show_idx=show_idx, text=show_labels)
        else:
            layout = draw_bboxes(layout, bbox, label,
                                 show_idx=show_idx, text=show_labels)

        fakes.append(layout.permute(1, 2, 0))

    # fixed noise vectors, sample them once outside of the loop
    if not 'z_obj' in variable:
        z_obj = torch.from_numpy(truncted_random(
            num_o=num_o, thres=thres)).float().cuda()
    if not 'z_img' in variable:
        z_im = torch.from_numpy(truncted_random(
            num_o=1, thres=thres)).view(1, -1).float().cuda()

    # varying inputs
    # use different once for each generated image
    for idx in range(num_gen):

        # if the layout is variable get a new one at each iteration
        if 'layout' in variable:
            _, label, bbox, depth, *_ = dataset[layout_idx+idx]

            if isinstance(bbox, np.ndarray):
                bbox = torch.from_numpy(bbox)

            # display layout by displaying bounding boxes on a black tensor
            layout = torch.zeros(3, 128, 128).type(torch.uint8)
            layout = draw_bboxes(layout, bbox, label, show_idx=show_idx)
            fakes.append(layout.permute(1, 2, 0))

        bbox_cp = bbox.clone()

        # transform some bounding boxes if required
        if not transform_bbox is None and idx > 0:
            for box_i in transform_bbox:
                xl, yl, w, h = bbox_cp[box_i]

                # if not transformation is provided, apply a random one
                if transform is None:
                    bbox_cp[box_i] = torch.tensor(
                        [xl+random.uniform(-0.5, 0.5), yl+random.uniform(-0.5, 0.5), w*random.uniform(0, 2), h*random.uniform(0, 2)])
                else:
                    bbox_cp = transform(box_i, idx, bbox_cp)

            # display layout by displaying bounding boxes on a black tensor
            # coloring the transformed ones in red
            layout = torch.zeros(3, 128, 128).type(torch.uint8)
            layout = draw_bboxes(
                layout, bbox_cp, label, color_transform=transform_bbox, show_idx=show_idx)
            fakes.append(layout.permute(1, 2, 0))

        # sample new noise vectors at each iteration
        if 'z_obj' in variable:
            z_obj = torch.from_numpy(truncted_random(
                num_o=num_o, thres=thres)).float().cuda()
        if 'z_img' in variable:
            z_im = torch.from_numpy(truncted_random(
                num_o=1, thres=thres)).view(1, -1).float().cuda()

        if use_depth:
            fake_images = netGdepth.forward(z_obj, bbox_cp.cuda().unsqueeze(
                0), z_im=z_im, y=label.long().cuda(), depths=depth.cuda().unsqueeze(0))
        else:
            fake_images = netGbase.forward(
                z_obj, bbox_cp.cuda().unsqueeze(0), z_im, label.long().cuda())

        # normalize from [-1,1] to [0,1]
        fakes.append(fake_images.detach().squeeze().permute(
            1, 2, 0).cpu() * 0.5 + 0.5)

    # put all images in a plot grid
    n_rows = int(math.ceil((len(fakes))/6))
    _, axs = plt.subplots(n_rows, max_cols, figsize=figsize)

    for idx, image in enumerate(fakes):
        row = idx // max_cols
        col = idx % max_cols
        # print(f'{row} {col}')
        if n_rows > 1:
            axs[row, col].axis("off")
            axs[row, col].imshow(image, aspect="auto")
        else:
            axs[col].axis("off")
            axs[col].imshow(image, aspect="auto")
    plt.subplots_adjust(wspace=.05, hspace=.05)
    plt.show()


def depth_transform(idx: int, frames: int = 2, transform_idx: list[int] = None, transform: Callable = None, transform_args=None):
    '''
    Displays the layout, original image, fake image generated by the baseline model, depthmap and the fake image generated by the depth-aware model.
    Then a transformation (can modify depth information) is applied to the bboxes in `transform_idx` and the depth-aware fake is generated again and displayed.

    Args:
        idx: layout index in the dataset
        frames: number of image generated by the layout
        transform_idx: indices of the bboxes to be transformed
        transform: function used to transform bboxes
        transform_args: additional arguments to be passed to the transform function
    '''

    # control plot order and size
    rows, cols, figsize = frames+1, 2, (8, 12)

    _, axs = plt.subplots(rows, cols, figsize=figsize)

    real, label, bbox, depth = dataset[idx]

    if isinstance(bbox, np.ndarray):
        bbox = torch.from_numpy(bbox)

    #### DISPLAY ORIGINAL ###

    # from [-1,1] to [0,255]
    real = ((real.cpu() + 1) / 2 * 255).type(torch.uint8)
    # # draw boxes
    # real = draw_bounding_boxes(real, coord_box)

    axs[0, 0].imshow(real.permute(1, 2, 0))
    axs[0, 0].set_title('Real image')
    axs[0, 0].axis('off')

    # sample new noise vectors at each iteration
    z_obj = torch.from_numpy(truncted_random(
        num_o=num_o, thres=thres)).float().cuda()
    z_im = torch.from_numpy(truncted_random(
        num_o=1, thres=thres)).view(1, -1).float().cuda()

    # generate fake
    fake = netGbase.forward(z_obj, bbox.clone().cuda().unsqueeze(
        0), z_im, label.long().cuda()).squeeze()
    axs[0, 1].imshow(fake.detach().permute(1, 2, 0).cpu()*0.5+0.5)
    axs[0, 1].set_title('Baseline fake image')
    axs[0, 1].axis('off')

    #### TRANSFORM AND GENERATE ####

    for t in range(frames):
        if transform_idx is not None and transform is not None:
            for box_i in transform_idx:
                if transform_args is None:
                    bbox, depth = transform(box_i, t, bbox, depth)
                else:
                    bbox, depth = transform(
                        box_i, t, bbox, depth, *transform_args)

        # t = 0 -> unmodified fake

        # new depth layout
        depth_layout = get_depth_layout(depth, real.shape[-2:], bbox)
        axs[t+1, 0].imshow(depth_layout, cmap='gray')
        if t == 0:
            axs[t+1, 0].set_title(f'Original depth layout')
        else:
            axs[t+1, 0].set_title(f'Modified depth layout {t}')

        axs[t+1, 0].axis('off')

        # new fake
        fake = netGdepth.forward(z_obj, bbox.clone().cuda().unsqueeze(
            0), z_im=z_im, y=label.long().cuda(), depths=depth.clone().cuda().unsqueeze(0)).squeeze()

        axs[t+1, 1].imshow(fake.detach().permute(1, 2, 0).cpu()*0.5+0.5)
        axs[t+1, 1].set_title(f'Depth-aware fake {t if t > 0 else ""}')
        axs[t+1, 1].axis('off')

        print(f'Fake {t+1} depths: {[(i, d) for i, d in enumerate(depth)]}')

    plt.show()


def move_objects_video(num_gen: int = 23, layout_idx: int = 0, transform_bbox: list[int] = None, transform: Callable = None, pixels: int = 1, vid_name: str = None):
    '''
    Given one layout, multiple frames are generated using both the baseline and the depth-aware model.
    In each frame some bounding boxes are moved/transformed.
    Generated frames are stacked on each other (baseline on top, depth-aware on bottom) and saved in a video for easier comparison.

    Args:
        num_gen: number of frames generated
        layout_idx: layout index in the dataset
        transform_bbox: list of indices indicating the bboxes to be transformed
        transform: function used to transform the bboxes indicated by `transform_bbox`
        pixels: displacement in pixels of the transformed objects at each frame
        vid_name: the video filename will be {dataset name}_{`vid_name`}_{`layout_idx`}.avi
    '''

    fakes = []

    z_obj = torch.from_numpy(truncted_random(
        num_o=num_o, thres=thres)).float().cuda()
    z_im = torch.from_numpy(truncted_random(
        num_o=1, thres=thres)).view(1, -1).float().cuda()

    prev_base = prev_depth = None

    for idx in tqdm(range(num_gen), 'Generating frames'):

        if idx == 0:
            _, label, bbox, depth, *_ = dataset[layout_idx]

            if isinstance(bbox, np.ndarray):
                bbox = torch.from_numpy(bbox)

            layout = torch.zeros(3, 128, 128).type(torch.uint8)
            layout = draw_bboxes(
                layout, bbox, label, text=False, show_idx=True, color_transform=transform_bbox)
            fakes.append(layout.permute(1, 2, 0))
            bbox_cp = bbox.clone()

        if not transform_bbox is None and idx > 0:
            for box_i in transform_bbox:
                xl, yl, w, h = bbox_cp[box_i]

                if transform is None:
                    # slightly move to the right
                    bbox_cp[box_i] = torch.tensor(
                        [xl+pixels*(1/128), yl, w, h])
                else:
                    bbox_cp = transform(box_i, idx, pixels, bbox_cp)

        fake_images_depth = netGdepth.forward(z_obj, bbox_cp.cuda().unsqueeze(
            0), z_im=z_im, y=label.long().cuda(), depths=depth.cuda().unsqueeze(0))

        fake_images_base = netGbase.forward(
            z_obj, bbox_cp.cuda().unsqueeze(0), z_im, label.long().cuda())

        # normalize from [-1,1] to [0,255]
        fake_images_base = (
            (fake_images_base.detach().squeeze().cpu() + 1) / 2 * 255).type(torch.uint8)
        fake_images_depth = (
            (fake_images_depth.detach().squeeze().cpu() + 1) / 2 * 255).type(torch.uint8)

        # difference between subsequent frames to highlight boundaries between objects
        if prev_base is None:
            # if it's the first frame, the first difference is set to zero
            timediff_base = timediff_depth = torch.zeros(
                fake_images_base.shape)
        else:
            timediff_base = T.rgb_to_grayscale(
                fake_images_base.clone(), num_output_channels=3) - prev_base
            timediff_depth = T.rgb_to_grayscale(
                fake_images_depth.clone(), num_output_channels=3) - prev_depth

        prev_base = T.rgb_to_grayscale(
            fake_images_base.clone(), num_output_channels=3)
        prev_depth = T.rgb_to_grayscale(
            fake_images_depth.clone(), num_output_channels=3)

        fake_edge_base = fake_images_base.clone().permute(1, 2, 0).numpy()
        fake_edge_depth = fake_images_depth.clone().permute(1, 2, 0).numpy()

        # compute Canny lower and upper thresholds based on median
        vb = np.median(fake_edge_base)
        vd = np.median(fake_edge_depth)
        sigma = 0.33

        lowerb = int(max(0, (1.0 - sigma) * vb))
        upperb = int(min(255, (1.0 + sigma) * vb))
        lowerd = int(max(0, (1.0 - sigma) * vd))
        upperd = int(min(255, (1.0 + sigma) * vd))

        # Canny edge detection to highlight boundaries between objects
        fake_edge_base = cv2.Canny(fake_edge_base, lowerb, upperb)
        fake_edge_depth = cv2.Canny(fake_edge_depth, lowerd, upperd)

        # from [0,255] to [0,1]
        # expand the tensor so it goes from single-channel image to a 3-channel one
        fake_edge_base = (torch.from_numpy(fake_edge_base) /
                          255).unsqueeze(0).expand(3, -1, -1)
        fake_edge_depth = (torch.from_numpy(fake_edge_depth) /
                           255).unsqueeze(0).expand(3, -1, -1)

        # add a text description
        c_red = (255,0,0)
        fake_images_base = draw_text(fake_images_base, 'baseline')
        fake_images_depth = draw_text(fake_images_depth, 'depth-aware')
        timediff_base = draw_text(timediff_base, 'timediff', c_red)
        timediff_depth = draw_text(timediff_depth, 'timediff', c_red)
        fake_edge_base = draw_text(fake_edge_base, 'edges', c_red)
        fake_edge_depth = draw_text(fake_edge_depth, 'edges', c_red)

        fake_images_base = torch.cat(
            (fake_images_base, timediff_base, fake_edge_base), dim=2)
        fake_images_depth = torch.cat(
            (fake_images_depth, timediff_depth, fake_edge_depth), dim=2)

        # normal on top, depth on bottom
        # tensors are now in [0,1]
        fake_images = torch.cat(
            (fake_images_base, fake_images_depth), dim=1).permute(1, 2, 0)

        # back to [0,255] uint8
        fakes.append(
            (fake_images*255).type(torch.uint8).numpy().astype(np.uint8))

    if not Path('samples/vids/'+ args.model_depth_name).is_dir():
        os.makedirs('samples/vids/'+ args.model_depth_name)

    h, w, c = fakes[1].shape
    filename = f'samples/vids/{args.model_depth_name}/{args.dataset}_{vid_name}_{layout_idx}.avi'

    fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
    writer = cv2.VideoWriter(filename, fourcc, 10, (w, h))

    print(f'Saving generated frames...')

    # save frames as a video
    for frame in fakes[1:]:
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        writer.write(frame)

    writer.release()

    print(f'Frames saved in {filename}')


def knn_analysis(use_depth: bool = True, figunitsize: int = 2, knn_k: int = 16, vis_knn: int = 10):
    '''
    For a ground truth image the function finds the `knn_k` nearest neighbors in the generated images.
    Then, for a generated image, the function finds the `knn_k` nearest neighbors in the ground truth images.
    The process is repeated for `vis_knn` query images.
    The distance is computed on the images' inception features.

    Args:
        use_depth: whether to use the depth-aware model or not
        figunitsize: dimension of a single image in the grid
        knn_k: number of neighbors to look for
        vis_knn: number of query images
    '''

    # output directory samples/dataset-model_name
    sample_path = os.path.join('samples', args.dataset + '-')

    fake_path = Path(
        sample_path + (args.model_depth_name if use_depth else args.model_name))

    if not fake_path.is_dir():
        if use_depth:
            sample_test(netGdepth, dataset, num_o, fake_path, 0)
        else:
            sample_test(netGbase, dataset_base, num_o, fake_path, 0)
    else:
        print(f'Found samples in {fake_path}')

    knn_dict = ev.knn_vis(dataset.image_dir, fake_path,
                          args.dataset, 128, batch_size=40, knn_k=knn_k, vis_knn=vis_knn)

    figsize = ((knn_k+1)*figunitsize, vis_knn*figunitsize)

    print('Query ground truths by sample')
    plt.figure(figsize=figsize)
    plt.imshow(make_grid([T.to_tensor(wandb_img.image)
               for wandb_img in knn_dict['knn_inception_query_gt_by_sample']], nrow=1).permute(1, 2, 0))
    plt.axis('off')
    plt.show()

    print('Query samples by ground truth')
    plt.figure(figsize=figsize)
    plt.imshow(make_grid([T.to_tensor(wandb_img.image)
               for wandb_img in knn_dict['knn_inception_query_sample_by_gt']], nrow=1).permute(1, 2, 0))
    plt.axis('off')
    plt.show()


def visualize_attention(idx: int = None, show_labels: bool = False):
    '''
    Generates a fake image with the depth-aware model and plots the attention map

    Args:
        idx: image position in the dataset. If none is specified a random one will be selected
        show_labels: if True the labels will be visualized on the layout
    '''

    # if no image is specified, select a random one
    idx = int(np.ceil(np.random.random()*len(dataset)) -
              1) if idx is None else idx
    print(f'Image: {idx}')

    # control plot order and size
    cols, figH = (3, 4)
    figsize = (figH*cols, figH)

    _, axs = plt.subplots(1, cols, figsize=figsize)

    real, labels, bbox, depth = dataset[idx]

    if isinstance(bbox, np.ndarray):
        bbox = torch.from_numpy(bbox)

    # get depth layout
    depth_layout = get_depth_layout(depth, real.shape[-2:], bbox)

    axs[0].imshow(depth_layout, cmap='gray')
    axs[0].set_title('Depth layout')

    # sample noise vectors
    z_obj = torch.from_numpy(truncted_random(
        num_o=num_o, thres=thres)).float().cuda()
    z_im = torch.from_numpy(truncted_random(
        num_o=1, thres=thres)).view(1, -1).float().cuda()

    # sample depth-aware
    fake, attn = netGdepth.forward(z_obj, bbox.clone().cuda().unsqueeze(
        0), z_im=z_im, y=labels.clone().long().cuda(), depths=depth.cuda().unsqueeze(0), return_attn=True)

    # normalize from [-1,1] to [0,1]
    fake = fake.detach().squeeze().permute(
        1, 2, 0).cpu() * 0.5 + 0.5
    
    axs[1].imshow(fake)
    axs[1].set_title('Fake image')

    # normalize from [-1,1] to [0,255]
    real = ((real.cpu() + 1) / 2 * 255).type(torch.uint8).permute(1,2,0)

    attnmap = attn.detach().squeeze().max(dim=-1).values.cpu().numpy()
    attnmap = pyramid_expand(attnmap[0], upscale=int(real.shape[0]/attnmap.shape[1]), sigma=8)

    axs[2].imshow(real)
    axs[2].set_title('Attention head 0')
    axs[2].imshow(attnmap, alpha=0.7, cmap='Greys_r')
    for ax in axs:
        ax.axis('off')
    plt.show()


class transforms:
    '''   Bounding boxes and depth transforms   '''

    @staticmethod
    def swap_depth_transform(box_i, t, bbox, depth, a, b):
        if t > 0:
            tmp = depth[a].clone()
            depth[a] = depth[b].clone()
            depth[b] = tmp

        return bbox, depth

    @staticmethod
    def move_down(box_i, idx, pixels, bbox):
        xl, yl, w, h = bbox[box_i]
        bbox[box_i] = torch.tensor([xl, yl+pixels*(1/128), w, h])
        return bbox

    @staticmethod
    def move_up(box_i, idx, pixels, bbox):
        xl, yl, w, h = bbox[box_i]
        bbox[box_i] = torch.tensor([xl, yl-pixels*(1/128), w, h])
        return bbox

    @staticmethod
    def move_left(box_i, idx, pixels, bbox):
        xl, yl, w, h = bbox[box_i]
        bbox[box_i] = torch.tensor([xl-pixels*(1/128), yl, w, h])
        return bbox

    @staticmethod
    def move_right(box_i, idx, pixels, bbox):
        xl, yl, w, h = bbox[box_i]
        bbox[box_i] = torch.tensor([xl+pixels*(1/128), yl, w, h])
        return bbox

    @staticmethod
    def move_left_bigger(box_i, idx, pixels, bbox):
        xl, yl, w, h = bbox[box_i]
        dh = 60/128 if idx == 0 else 0
        bbox[box_i] = torch.tensor([xl-pixels*(1/128), yl, w, h+dh])
        return bbox

    @classmethod
    def right_diag(Q, box_i, idx, pixels, bbox):
        bbox = Q.move_up(box_i, idx, pixels, bbox)
        bbox = Q.move_right(box_i, idx, pixels, bbox)
        return bbox
