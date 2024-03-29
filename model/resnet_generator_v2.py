import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb

from utils.depth import get_depth_layout_batch
from .norm_module import *
from .mask_regression import *
from .sync_batchnorm import SynchronizedBatchNorm2d
from .attention import TransformerBlock

BatchNorm = SynchronizedBatchNorm2d


class ResnetGenerator128(nn.Module):
    def __init__(self, ch=64, z_dim=128, num_classes=10, output_dim=3):
        super(ResnetGenerator128, self).__init__()
        self.num_classes = num_classes

        self.label_embedding = nn.Embedding(num_classes, 180)

        num_w = 128+180
        self.fc = nn.utils.spectral_norm(nn.Linear(z_dim, 4*4*16*ch))

        self.res1 = ResBlock(ch*16, ch*16, upsample=True, num_w=num_w)
        self.res2 = ResBlock(ch*16, ch*8, upsample=True, num_w=num_w)
        self.res3 = ResBlock(ch*8, ch*4, upsample=True, num_w=num_w)
        self.res4 = ResBlock(ch*4, ch*2, upsample=True, num_w=num_w, psp_module=True)
        self.res5 = ResBlock(ch*2, ch*1, upsample=True, num_w=num_w, predict_mask=False)
        self.final = nn.Sequential(BatchNorm(ch),
                                   nn.ReLU(),
                                   conv2d(ch, output_dim, 3, 1, 1),
                                   nn.Tanh())

        # mapping function
        mapping = list()
        self.mapping = nn.Sequential(*mapping)

        self.alpha1 = nn.Parameter(torch.zeros(1, 184, 1))
        self.alpha2 = nn.Parameter(torch.zeros(1, 184, 1))
        self.alpha3 = nn.Parameter(torch.zeros(1, 184, 1))
        self.alpha4 = nn.Parameter(torch.zeros(1, 184, 1))

        self.sigmoid = nn.Sigmoid()

        self.mask_regress = MaskRegressNetv2(num_w)
        self.init_parameter()

    def forward(self, z, bbox, z_im=None, y=None) -> torch.Tensor:
        # z shape: batch, number of objects, latent code length

        # b = batch size, o = number of objects/latents
        b, o = z.size(0), z.size(1)
        label_embedding = self.label_embedding(y)

        z = z.view(b * o, -1)
        label_embedding = label_embedding.view(b * o, -1)

        latent_vector = torch.cat((z, label_embedding), dim=1).view(b, o, -1)

        w = self.mapping(latent_vector.view(b * o, -1))
        
        # preprocess bbox
        # predict a 64x64 mask for each object in each image
        # each mask has 0 outside of the corresponding bounding box
        # M_s in the paper
        # (batch, num_o, 64, 64)
        bmask = self.mask_regress(w, bbox)

        if z_im is None:
            z_im = torch.randn((b, 128), device=z.device)

        # 64x64 binary mask for each bounding box
        # 1 inside bbox, 0 outside
        # used to clip the mask obtained from
        # each resblock output features
        # (batch, num_o, 64, 64)
        bbox_mask_ = bbox_mask(z, bbox, 64, 64)

        # 4x4
        # (batch, 1024, 4, 4)
        x = self.fc(z_im).view(b, -1, 4, 4)

        # 8x8
        # each resblock returns a mask using
        # a small convolutional network on it's own output features
        # (batch, 1024, 8, 8), (batch, 184, 8, 8)
        x, stage_mask = self.res1(x, w, bmask)

        # 16x16
        hh, ww = x.size(2), x.size(3)

        # from stage_mask (batch, 184, h, w) get only 31 masks
        # resulting in (batch, num_o, h, w), a mask for each obj
        # (batch, num_o, 8, 8)
        seman_bbox = batched_index_select(stage_mask, dim=1, index=y.view(b, o, 1, 1)) # size (b, num_o, h, w)

        # clip the masks so that everything outside the corresponding
        # bounding box is set to 0
        # M_f in the paper
        # (batch, num_o, 8, 8)
        seman_bbox = torch.sigmoid(seman_bbox) * F.interpolate(bbox_mask_, size=(hh, ww), mode='nearest')
        
        # 184 is the max number of possible classes
        # alpha learnable weight to balance the two masks
        # expand alpha from (1, 184, 1) to (batch, 184, 1)
        # reshape labels y from (batch, num_o) to (batch, num_o, 1) to match alpha
        # from the 184 alphas extract the relevant num_o ones, one alpha for each label
        # unsqueeze alpha to append a dimension at the end
        # (batch, num_o, 1, 1)
        alpha1 = torch.gather(self.sigmoid(self.alpha1).expand(b, -1, -1), dim=1, index=y.view(b, o, 1)).unsqueeze(-1)

        # bmask: masks predicted by regressor from the bbox
        # seman_bbox: masks predicted by the convolutional ToMask operation
        #             in the ResBlock, using the output features of the block itself
        #             clipped based on the layout, 0 outside of the bboxes
        # combine the two masks using learnable alpha as weight
        # M_s [bmask] * (1 - alpha) + M_f [seman_bbox] * alpha
        # (batch, num_o, 8, 8)
        stage_bbox = F.interpolate(bmask, size=(hh, ww), mode='bilinear') * (1 - alpha1) + seman_bbox * alpha1

        # (batch, 512, 16, 16), (batch, 184, 16, 16)
        x, stage_mask = self.res2(x, w, stage_bbox)

        # 32x32
        hh, ww = x.size(2), x.size(3)
        seman_bbox = batched_index_select(stage_mask, dim=1, index=y.view(b, o, 1, 1)) # size (b, num_o, h, w)
        seman_bbox = torch.sigmoid(seman_bbox) * F.interpolate(bbox_mask_, size=(hh, ww), mode='nearest')
        alpha2 = torch.gather(self.sigmoid(self.alpha2).expand(b, -1, -1), dim=1, index=y.view(b, o, 1)).unsqueeze(-1) 
        stage_bbox = F.interpolate(bmask, size=(hh, ww), mode='bilinear') * (1 - alpha2) + seman_bbox * alpha2
        x, stage_mask = self.res3(x, w, stage_bbox)

        # 64x64
        hh, ww = x.size(2), x.size(3)
        seman_bbox = batched_index_select(stage_mask, dim=1, index=y.view(b, o, 1, 1)) # size (b, num_o, h, w)
        seman_bbox = torch.sigmoid(seman_bbox) * F.interpolate(bbox_mask_, size=(hh, ww), mode='nearest')
        alpha3 = torch.gather(self.sigmoid(self.alpha3).expand(b, -1, -1), dim=1, index=y.view(b, o, 1)).unsqueeze(-1) 
        stage_bbox = F.interpolate(bmask, size=(hh, ww), mode='bilinear') * (1 - alpha3) + seman_bbox * alpha3
        x, stage_mask = self.res4(x, w, stage_bbox)

        # 128x128
        hh, ww = x.size(2), x.size(3)
        seman_bbox = batched_index_select(stage_mask, dim=1, index=y.view(b, o, 1, 1)) # size (b, num_o, h, w)
        seman_bbox = torch.sigmoid(seman_bbox) * F.interpolate(bbox_mask_, size=(hh, ww), mode='nearest')
        alpha4 = torch.gather(self.sigmoid(self.alpha4).expand(b, -1, -1), dim=1, index=y.view(b, o, 1)).unsqueeze(-1) 
        stage_bbox = F.interpolate(bmask, size=(hh, ww), mode='bilinear') * (1 - alpha4) + seman_bbox * alpha4
        x, _ = self.res5(x, w, stage_bbox)

        # to RGB
        x = self.final(x)
        return x

    def init_parameter(self):
        for k in self.named_parameters():
            if k[1].dim() > 1:
                torch.nn.init.orthogonal_(k[1])
            if k[0][-4:] == 'bias':
                torch.nn.init.constant_(k[1], 0)


class ResnetGeneratorTransfFeats128(nn.Module):
    def __init__(self, ch=64, z_dim=128, num_classes=10, num_o=20, output_dim=3, attntype='default'):
        super(ResnetGeneratorTransfFeats128, self).__init__()
        self.num_classes = num_classes

        self.label_embedding = nn.Embedding(num_classes, 180)

        num_w = 128+180


        # attention type
        self.attntype = attntype
        context_dim, crossonly = {
            'default': (1, False),
            'selfonly': (None, False),
            'crossonly': (1, True)
        }[attntype]

        # transformer blocks parameters
        transf_depth = 1
        num_heads = 8
        dim_heads = 64

        if wandb.run is not None:
            wandb.config.update({'attention_heads': num_heads, 'heads_dim': dim_heads, 'transf_depth': transf_depth})

        self.fc = nn.utils.spectral_norm(nn.Linear(z_dim, 4*4*16*ch))

        self.res1 = ResBlock(ch*16, ch*16, upsample=True, num_w=num_w)
        self.res2 = ResBlock(ch*16, ch*8, upsample=True, num_w=num_w)
        self.res3 = ResBlock(ch*8, ch*4, upsample=True, num_w=num_w)
        self.res4 = ResBlock(ch*4, ch*2, upsample=True, num_w=num_w, psp_module=True)
        self.res5 = ResBlock(ch*2, ch*1, upsample=True, num_w=num_w, predict_mask=False)

        self.transf3 = TransformerBlock(dim=ch*4, context_dim=context_dim, heads=num_heads, dim_head=dim_heads, depth=transf_depth, crossonly=crossonly)

        self.final = nn.Sequential(BatchNorm(ch),
                                   nn.ReLU(),
                                   conv2d(ch, output_dim, 3, 1, 1),
                                   nn.Tanh())

        # mapping function
        mapping = list()
        self.mapping = nn.Sequential(*mapping)

        self.alpha1 = nn.Parameter(torch.zeros(1, 184, 1))
        self.alpha2 = nn.Parameter(torch.zeros(1, 184, 1))
        self.alpha3 = nn.Parameter(torch.zeros(1, 184, 1))
        self.alpha4 = nn.Parameter(torch.zeros(1, 184, 1))

        self.sigmoid = nn.Sigmoid()

        self.mask_regress = MaskRegressNetv2(num_w)
        self.init_parameter()

    def forward(self, z, bbox, y, depths, z_im=None, return_attn=False) -> torch.Tensor:
        # z shape: batch, number of objects, latent code length

        # b = batch size, o = number of objects/latents
        b, o = z.size(0), z.size(1)
        label_embedding = self.label_embedding(y)

        z = z.view(b * o, -1)
        label_embedding = label_embedding.view(b * o, -1)

        latent_vector = torch.cat((z, label_embedding), dim=1).view(b, o, -1)

        w = self.mapping(latent_vector.view(b * o, -1))

        depths = depths.unsqueeze(-1)
        
        # preprocess bbox
        # predict a 64x64 mask for each object in each image
        # each mask has 0 outside of the corresponding bounding box
        # M_s in the paper
        # (batch, num_o, 64, 64)
        bmask = self.mask_regress(w, bbox)

        if z_im is None:
            z_im = torch.randn((b, 128), device=z.device)

        # 64x64 binary mask for each bounding box
        # 1 inside bbox, 0 outside
        # used to clip the mask obtained from
        # each resblock output features
        # (batch, num_o, 64, 64)
        bbox_mask_ = bbox_mask(z, bbox, 64, 64)

        # 4x4
        # (batch, 1024, 4, 4)
        x = self.fc(z_im).view(b, -1, 4, 4)

        # 8x8
        # each resblock returns a mask using
        # a small convolutional network on it's own output features
        # (batch, 1024, 8, 8), (batch, 184, 8, 8)
        x, stage_mask = self.res1(x, w, bmask)

        # 16x16
        hh, ww = x.size(2), x.size(3)

        # from stage_mask (batch, 184, h, w) get only 31 masks
        # resulting in (batch, num_o, h, w), a mask for each obj
        # (batch, num_o, 8, 8)
        seman_bbox = batched_index_select(stage_mask, dim=1, index=y.view(b, o, 1, 1)) # size (b, num_o, h, w)

        # clip the masks so that everything outside the corresponding
        # bounding box is set to 0
        # M_f in the paper
        # (batch, num_o, 8, 8)
        seman_bbox = torch.sigmoid(seman_bbox) * F.interpolate(bbox_mask_, size=(hh, ww), mode='nearest')
        
        # 184 is the max number of possible classes
        # alpha learnable weight to balance the two masks
        # expand alpha from (1, 184, 1) to (batch, 184, 1)
        # reshape labels y from (batch, num_o) to (batch, num_o, 1) to match alpha
        # from the 184 alphas extract the relevant num_o ones, one alpha for each label
        # unsqueeze alpha to append a dimension at the end
        # (batch, num_o, 1, 1)
        alpha1 = torch.gather(self.sigmoid(self.alpha1).expand(b, -1, -1), dim=1, index=y.view(b, o, 1)).unsqueeze(-1)

        # bmask: masks predicted by regressor from the bbox
        # seman_bbox: masks predicted by the convolutional ToMask operation
        #             in the ResBlock, using the output features of the block itself
        #             clipped based on the layout, 0 outside of the bboxes
        # combine the two masks using learnable alpha as weight
        # M_s [bmask] * (1 - alpha) + M_f [seman_bbox] * alpha
        # (batch, num_o, 8, 8)
        stage_bbox = F.interpolate(bmask, size=(hh, ww), mode='bilinear') * (1 - alpha1) + seman_bbox * alpha1

        # (batch, 512, 16, 16), (batch, 184, 16, 16)
        x, stage_mask = self.res2(x, w, stage_bbox)

        # 32x32
        hh, ww = x.size(2), x.size(3)
        seman_bbox = batched_index_select(stage_mask, dim=1, index=y.view(b, o, 1, 1)) # size (b, num_o, h, w)
        seman_bbox = torch.sigmoid(seman_bbox) * F.interpolate(bbox_mask_, size=(hh, ww), mode='nearest')
        alpha2 = torch.gather(self.sigmoid(self.alpha2).expand(b, -1, -1), dim=1, index=y.view(b, o, 1)).unsqueeze(-1) 
        stage_bbox = F.interpolate(bmask, size=(hh, ww), mode='bilinear') * (1 - alpha2) + seman_bbox * alpha2

        x, stage_mask = self.res3(x, w, stage_bbox)

        # 64x64
        hh, ww = x.size(2), x.size(3)
        seman_bbox = batched_index_select(stage_mask, dim=1, index=y.view(b, o, 1, 1)) # size (b, num_o, h, w)
        seman_bbox = torch.sigmoid(seman_bbox) * F.interpolate(bbox_mask_, size=(hh, ww), mode='nearest')
        alpha3 = torch.gather(self.sigmoid(self.alpha3).expand(b, -1, -1), dim=1, index=y.view(b, o, 1)).unsqueeze(-1) 
        stage_bbox = F.interpolate(bmask, size=(hh, ww), mode='bilinear') * (1 - alpha3) + seman_bbox * alpha3

        # (batch, c, h, w)
        # apply feats-depth attention
        x, attn = self.transf3(x, context=depths.clone(), return_attn=True)

        x, stage_mask = self.res4(x, w, stage_bbox)

        # 128x128
        hh, ww = x.size(2), x.size(3)
        seman_bbox = batched_index_select(stage_mask, dim=1, index=y.view(b, o, 1, 1)) # size (b, num_o, h, w)
        seman_bbox = torch.sigmoid(seman_bbox) * F.interpolate(bbox_mask_, size=(hh, ww), mode='nearest')
        alpha4 = torch.gather(self.sigmoid(self.alpha4).expand(b, -1, -1), dim=1, index=y.view(b, o, 1)).unsqueeze(-1) 
        stage_bbox = F.interpolate(bmask, size=(hh, ww), mode='bilinear') * (1 - alpha4) + seman_bbox * alpha4
        
        x, _ = self.res5(x, w, stage_bbox)


        # to RGB
        x = self.final(x)

        if return_attn:
            return x, attn
        return x

    def init_parameter(self):
        for k in self.named_parameters():
            if k[1].dim() > 1:
                torch.nn.init.orthogonal_(k[1])
            if k[0][-4:] == 'bias':
                torch.nn.init.constant_(k[1], 0)


class ResnetGenerator256(nn.Module):
    def __init__(self, ch=64, z_dim=128, num_classes=10, output_dim=3):
        super(ResnetGenerator256, self).__init__()
        self.num_classes = num_classes

        self.label_embedding = nn.Embedding(num_classes, 180)

        num_w = 128+180
        self.fc = nn.utils.spectral_norm(nn.Linear(z_dim, 4*4*16*ch))

        self.res1 = ResBlock(ch*16, ch*16, upsample=True, num_w=num_w)
        self.res2 = ResBlock(ch*16, ch*8, upsample=True, num_w=num_w)
        self.res3 = ResBlock(ch*8, ch*8, upsample=True, num_w=num_w)
        self.res4 = ResBlock(ch*8, ch*4, upsample=True, num_w=num_w)
        self.res5 = ResBlock(ch*4, ch*2, upsample=True, num_w=num_w)
        self.res6 = ResBlock(ch*2, ch*1, upsample=True, num_w=num_w, predict_mask=False)
        self.final = nn.Sequential(BatchNorm(ch),
                                   nn.ReLU(),
                                   conv2d(ch, output_dim, 3, 1, 1),
                                   nn.Tanh())

        # mapping function
        mapping = list()
        self.mapping = nn.Sequential(*mapping)

        self.alpha1 = nn.Parameter(torch.zeros(1, 184, 1))
        self.alpha2 = nn.Parameter(torch.zeros(1, 184, 1))
        self.alpha3 = nn.Parameter(torch.zeros(1, 184, 1))
        self.alpha4 = nn.Parameter(torch.zeros(1, 184, 1))
        self.alpha5 = nn.Parameter(torch.zeros(1, 184, 1))
        self.sigmoid = nn.Sigmoid()

        self.mask_regress = MaskRegressNetv2(num_w)
        self.init_parameter()

    def forward(self, z, bbox, z_im=None, y=None, include_mask_loss=False):
        b, o = z.size(0), z.size(1)

        label_embedding = self.label_embedding(y)

        z = z.view(b * o, -1)
        label_embedding = label_embedding.view(b * o, -1)

        latent_vector = torch.cat((z, label_embedding), dim=1).view(b, o, -1)

        w = self.mapping(latent_vector.view(b * o, -1))

        # preprocess bbox
        bmask = self.mask_regress(w, bbox)

        if z_im is None:
            z_im = torch.randn((b, 128), device=z.device)

        bbox_mask_ = bbox_mask(z, bbox, 128, 128)
        
        latent_vector = torch.cat((z, label_embedding), dim=1).view(b, o, -1)
        w = self.mapping(latent_vector.view(b * o, -1))

        # 4x4
        x = self.fc(z_im).view(b, -1, 4, 4)
        # 8x8
        # label mask
        x, stage_mask = self.res1(x, w, bmask)

        # 16x16
        hh, ww = x.size(2), x.size(3)
        seman_bbox = batched_index_select(stage_mask, dim=1, index=y.view(b, o, 1, 1)) # size (b, num_o, h, w)
        seman_bbox = torch.sigmoid(seman_bbox) * F.interpolate(bbox_mask_, size=(hh, ww), mode='nearest')
        alpha1 = torch.gather(self.sigmoid(self.alpha1).expand(b, -1, -1), dim=1, index=y.view(b, o, 1)).unsqueeze(-1)
        stage_bbox = F.interpolate(bmask, size=(hh, ww), mode='bilinear') * (1 - alpha1) + seman_bbox * alpha1
        x, stage_mask = self.res2(x, w, stage_bbox)

        # 32x32
        hh, ww = x.size(2), x.size(3)
        seman_bbox = batched_index_select(stage_mask, dim=1, index=y.view(b, o, 1, 1)) # size (b, num_o, h, w)
        seman_bbox = torch.sigmoid(seman_bbox) * F.interpolate(bbox_mask_, size=(hh, ww), mode='nearest')

        alpha2 = torch.gather(self.sigmoid(self.alpha2).expand(b, -1, -1), dim=1, index=y.view(b, o, 1)).unsqueeze(-1)
        stage_bbox = F.interpolate(bmask, size=(hh, ww), mode='bilinear') * (1 - alpha2) + seman_bbox * alpha2
        x, stage_mask = self.res3(x, w, stage_bbox)

        # 64x64
        hh, ww = x.size(2), x.size(3)
        seman_bbox = batched_index_select(stage_mask, dim=1, index=y.view(b, o, 1, 1)) # size (b, num_o, h, w)
        seman_bbox = torch.sigmoid(seman_bbox) * F.interpolate(bbox_mask_, size=(hh, ww), mode='nearest')

        alpha3 = torch.gather(self.sigmoid(self.alpha3).expand(b, -1, -1), dim=1, index=y.view(b, o, 1)).unsqueeze(-1)
        stage_bbox = F.interpolate(bmask, size=(hh, ww), mode='bilinear') * (1 - alpha3) + seman_bbox * alpha3
        x, stage_mask = self.res4(x, w, stage_bbox)

        # 128x128
        hh, ww = x.size(2), x.size(3)
        seman_bbox = batched_index_select(stage_mask, dim=1, index=y.view(b, o, 1, 1)) # size (b, num_o, h, w)
        seman_bbox = torch.sigmoid(seman_bbox) * F.interpolate(bbox_mask_, size=(hh, ww), mode='nearest')

        alpha4 = torch.gather(self.sigmoid(self.alpha4).expand(b, -1, -1), dim=1, index=y.view(b, o, 1)).unsqueeze(-1)
        stage_bbox = F.interpolate(bmask, size=(hh, ww), mode='bilinear') * (1 - alpha4) + seman_bbox * alpha4
        x, stage_mask = self.res5(x, w, stage_bbox)

        # 256x256
        hh, ww = x.size(2), x.size(3)
        seman_bbox = batched_index_select(stage_mask, dim=1, index=y.view(b, o, 1, 1)) # size (b, num_o, h, w)
        seman_bbox = torch.sigmoid(seman_bbox) * F.interpolate(bbox_mask_, size=(hh, ww), mode='nearest')

        alpha5 = torch.gather(self.sigmoid(self.alpha5).expand(b, -1, -1), dim=1, index=y.view(b, o, 1)).unsqueeze(-1)
        stage_bbox = F.interpolate(bmask, size=(hh, ww), mode='bilinear') * (1 - alpha5) + seman_bbox * alpha5
        x, _ = self.res6(x, w, stage_bbox)
        # to RGB
        x = self.final(x)
        return x

    def init_parameter(self):
        for k in self.named_parameters():
            if k[1].dim() > 1:
                torch.nn.init.orthogonal_(k[1])
            if k[0][-4:] == 'bias':
                torch.nn.init.constant_(k[1], 0)


class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, h_ch=None, ksize=3, pad=1, upsample=False, num_w=128, predict_mask=True, psp_module=False):
        super(ResBlock, self).__init__()
        self.upsample = upsample
        self.h_ch = h_ch if h_ch else out_ch
        self.conv1 = conv2d(in_ch, self.h_ch, ksize, pad=pad)
        self.conv2 = conv2d(self.h_ch, out_ch, ksize, pad=pad)
        self.b1 = SpatialAdaptiveSynBatchNorm2d(in_ch, num_w=num_w, batchnorm_func=BatchNorm)
        self.b2 = SpatialAdaptiveSynBatchNorm2d(self.h_ch, num_w=num_w, batchnorm_func=BatchNorm)
        self.learnable_sc = in_ch != out_ch or upsample
        if self.learnable_sc:
            self.c_sc = conv2d(in_ch, out_ch, 1, 1, 0)
        self.activation = nn.ReLU()

        self.predict_mask = predict_mask
        if self.predict_mask:
            if psp_module:
                self.conv_mask = nn.Sequential(PSPModule(out_ch, 100),
                                               nn.Conv2d(100, 184, kernel_size=1))
            else:
                self.conv_mask = nn.Sequential(nn.Conv2d(out_ch, 100, 3, 1, 1),
                                               BatchNorm(100),
                                               nn.ReLU(),
                                               nn.Conv2d(100, 184, 1, 1, 0, bias=True))

    def residual(self, in_feat, w, bbox):
        x = in_feat
        x = self.b1(x, w, bbox)
        x = self.activation(x)
        if self.upsample:
            x = F.interpolate(x, scale_factor=2, mode='nearest')
        x = self.conv1(x)
        x = self.b2(x, w, bbox)
        x = self.activation(x)
        x = self.conv2(x)
        return x

    def shortcut(self, x):
        if self.learnable_sc:
            if self.upsample:
                x = F.interpolate(x, scale_factor=2, mode='nearest')
            x = self.c_sc(x)
        return x

    def forward(self, in_feat, w, bbox):
        # bbox is the resulting mask from the previous stage, M
        # the combination of the mask predicred by the mask regressor
        # and the clipped mask predicted by the previous ResBlock+ conv ToMask
        # not the bboxes coordinates

        out_feat = self.residual(in_feat, w, bbox) + self.shortcut(in_feat)
        
        # ToMask operation in the paper
        # + sigmoid later in the generator's forward
        # learns a mask from the output features of the resblock
        if self.predict_mask:
            mask = self.conv_mask(out_feat)
        else:
            mask = None
        return out_feat, mask


def conv2d(in_feat, out_feat, kernel_size=3, stride=1, pad=1, spectral_norm=True):
    conv = nn.Conv2d(in_feat, out_feat, kernel_size, stride, pad)
    if spectral_norm:
        return nn.utils.spectral_norm(conv, eps=1e-4)
    else:
        return conv


def batched_index_select(input, dim, index):
    # list of stage mask dims
    # [batch, 184, h, w]
    expanse = list(input.shape)
    # dims marked with -1 will be ignored by torch expand
    # ignore batch and the specified dim (1)
    expanse[0] = -1
    expanse[dim] = -1
    # expand labels from (batch, num_o, 1, 1)
    # to (batch, num_o, h, w) repeating values
    index = index.expand(expanse)
    # gets from input all elements specified by index
    # along dim (1) dimension
    # in this case from 184 it will take only num_o hxw tensors
    return torch.gather(input, dim, index)


def bbox_mask(x, bbox, H, W):
    b, o, _ = bbox.size()
    N = b * o

    bbox_1 = bbox.float().view(-1, 4)
    x0, y0 = bbox_1[:, 0], bbox_1[:, 1]
    ww, hh = bbox_1[:, 2], bbox_1[:, 3]

    x0 = x0.contiguous().view(N, 1).expand(N, H)
    ww = ww.contiguous().view(N, 1).expand(N, H)
    y0 = y0.contiguous().view(N, 1).expand(N, W)
    hh = hh.contiguous().view(N, 1).expand(N, W)

    X = torch.linspace(0, 1, steps=W).view(1, W).expand(N, W).cuda(device=x.device)
    Y = torch.linspace(0, 1, steps=H).view(1, H).expand(N, H).cuda(device=x.device)

    X = (X - x0) / ww
    Y = (Y - y0) / hh

    X_out_mask = ((X < 0) + (X > 1)).view(N, 1, W).expand(N, H, W)
    Y_out_mask = ((Y < 0) + (Y > 1)).view(N, H, 1).expand(N, H, W)

    out_mask = 1 - (X_out_mask + Y_out_mask).float().clamp(max=1)
    return out_mask.view(b, o, H, W)


class PSPModule(nn.Module):
    """
    Reference:
        Zhao, Hengshuang, et al. *"Pyramid scene parsing network."*
    """
    def __init__(self, features, out_features=512, sizes=(1, 2, 3, 6)):
        super(PSPModule, self).__init__()

        self.stages = []
        self.stages = nn.ModuleList([self._make_stage(features, out_features, size) for size in sizes])
        self.bottleneck = nn.Sequential(
            nn.Conv2d(features+len(sizes)*out_features, out_features, kernel_size=3, padding=1, dilation=1, bias=False),
            BatchNorm(out_features),
            nn.ReLU(),
            nn.Dropout2d(0.1)
            )

    def _make_stage(self, features, out_features, size):
        prior = nn.AdaptiveAvgPool2d(output_size=(size, size))
        conv = nn.Conv2d(features, out_features, kernel_size=1, bias=False)
        bn = nn.BatchNorm2d(out_features)
        return nn.Sequential(prior, conv, bn, nn.ReLU())

    def forward(self, feats):
        h, w = feats.size(2), feats.size(3)
        priors = [F.interpolate(input=stage(feats), size=(h, w), mode='bilinear', align_corners=True) for stage in self.stages] + [feats]
        bottle = self.bottleneck(torch.cat(priors, 1))
        return bottle
