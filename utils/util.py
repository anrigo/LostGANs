from typing import Union
import numpy as np
import torch
import torch.nn as nn
from torchvision import models
import torchvision.transforms.functional as T
import torch.nn.functional as F
from PIL.Image import Image
from PIL import ImageDraw


def crop_resize(image, bbox, imsize=64, cropsize=28, label=None):
    """"
    :param image: (b, 3, h, w)
    :param bbox: (b, o, 4)
    :param imsize: input image size
    :param cropsize: image size after crop
    :param label:
    :return: crop_images: (b*o, 3, h, w)
    """
    crop_images = list()
    b, o, _ = bbox.size()
    if label is not None:
        rlabel = list()
    for idx in range(b):
        for odx in range(o):
            if torch.min(bbox[idx, odx]) < 0:
                continue
            crop_image = image[idx:idx+1, :, int(imsize*bbox[idx, odx, 1]):int(imsize*(bbox[idx, odx, 1]+bbox[idx, odx, 3])),
                               int(imsize*bbox[idx, odx, 0]):int(imsize*(bbox[idx, odx, 0]+bbox[idx, odx, 2]))]
            crop_image = F.interpolate(crop_image, size=(
                cropsize, cropsize), mode='bilinear')
            crop_images.append(crop_image)
            if label is not None:
                rlabel.append(label[idx, odx, :].unsqueeze(0))
    # print(rlabel)
    if label is not None:
        # if len(rlabel) % 2 == 1:
        #    return torch.cat(crop_images[:-1], dim=0), torch.cat(rlabel[:-1], dim=0)
        return torch.cat(crop_images, dim=0), torch.cat(rlabel, dim=0)
    return torch.cat(crop_images, dim=0)


def truncted_random(num_o=8, thres=1.0):
    z = np.ones((1, num_o, 128)) * 100
    for i in range(num_o):
        for j in range(128):
            while z[0, i, j] > thres or z[0, i, j] < - thres:
                z[0, i, j] = np.random.normal()
    return z


# VGG Features matching
class Vgg19(torch.nn.Module):
    def __init__(self, requires_grad=False):
        super(Vgg19, self).__init__()
        vgg_pretrained_features = models.vgg19(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h_relu1 = self.slice1(X)
        h_relu2 = self.slice2(h_relu1)
        h_relu3 = self.slice3(h_relu2)
        h_relu4 = self.slice4(h_relu3)
        h_relu5 = self.slice5(h_relu4)
        out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
        return out


class VGGLoss(nn.Module):
    def __init__(self):
        super(VGGLoss, self).__init__()
        self.vgg = Vgg19().cuda()
        self.criterion = nn.L1Loss()
        self.weights = [1.0 / 32, 1.0 / 16, 1.0 / 8, 1.0 / 4, 1.0]

    def forward(self, x, y):
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)
        loss = 0

        for i in range(len(x_vgg)):
            loss += self.weights[i] * \
                self.criterion(x_vgg[i], y_vgg[i].detach())

        return loss


def normalize_tensor(tensor: torch.Tensor, to_: 'tuple[float, float]', from_: 'tuple[float, float]' = None, eps: float = 1e-12) -> torch.Tensor:
    '''
    Normalize tensor to the given range using min-max normalization (rescaling)

    Args:
        tensor: data tensor
        to: tuple (min, max) target range
        from: tuple (min, max) input data range, if not specified max and min of the input will be used
        eps: small constant to avoid division by zero
    '''

    min_, max_ = (tensor.min(), tensor.max()) if from_ is None else from_
    newmin, newmax = to_
    newmin, newmax = float(newmin), float(newmax)
    num = (tensor - min_) * (newmax - newmin)
    den = max((max_ - min_), eps)
    return (num / den) + newmin


def scale_boxes(boxes: torch.Tensor, shape: 'tuple[int, int]', format: str = None, dtype: torch.dtype = None) -> torch.Tensor:
    '''
    Scales bounding boxes to match the given image size

    Args:
        boxes: Tensor of bounding boxes in the (x, y, w, h) format, in the range (0,1)
        shape: tuple (height, width)
        format: 'coordinates' for (xmin, ymin, xmax, ymax) format, default format,
                'size' for (x, y, w, h) format,
                'inverse_size' for (y, x, h, w), compatible with torchvision crop utility function
        dtype: if specified the output tensor will be converted to this type, for example torch.int
    Returns:
        boxes: Tensor of bounding boxes in the specified format, scaled up to the specified shape
    '''
    bboxes = boxes.clone()

    for i, box in enumerate(bboxes):
        x, y, w, h = box
        hh, ww = shape

        if format is None or format == 'coordinates':
            # (xmin, ymin, xmax, ymax)
            bboxes[i] = torch.tensor(
                (int(x*ww), int(y*hh), int(x*ww)+int(w*ww), int(y*hh)+int(h*hh)))
        elif format == 'size':
            # (x, y, w, h)
            bboxes[i] = torch.tensor(
                (int(x*ww), int(y*hh), int(w*ww), int(h*hh)))
        elif format == 'inverse_size':
            # (y, x, h, w)
            bboxes[i] = torch.tensor(
                (int(y*hh), int(x*ww), int(h*hh), int(w*ww)))
        else:
            raise ValueError('Unrecognized format')

        if dtype is not None:
            bboxes = bboxes.type(dtype)

    return bboxes


def draw_text(img: Union[torch.Tensor, Image], text: str, color: tuple[int, int, int] = (0, 0, 0), location: tuple[int, int] = (0, 0)) -> Union[torch.Tensor, Image]:
    '''
    Draws text on an image

    Args:
        img: image
        text: text to draw
        color: color of the text
        location: where the top-left corner of the text will be in the image
    Returns:
        img: Image with the desired text, of the same type of the input image. If the input image was a Tensor, the output image will be a Tensor in the range [0,1]
    '''

    is_tensor = isinstance(img, torch.Tensor)
    if is_tensor:
        img = T.to_pil_image(img)

    draw = ImageDraw.Draw(img)
    draw.text(location, text, color)

    if is_tensor:
        img = T.to_tensor(img)

    return img


def count_parameters(model: nn.Module):
    '''Returns the number of parameters of a model'''
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
