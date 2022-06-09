import torch
import torch.nn as nn
import torch.nn.functional as F


# Adaptive instance normalization
# modified from https://github.com/NVlabs/MUNIT/blob/d79d62d99b588ae341f9826799980ae7298da553/networks.py#L453-L482
class AdaptiveInstanceNorm2d(nn.Module):
    def __init__(self, num_features, num_w=512, eps=1e-5, momentum=0.1):
        super(AdaptiveInstanceNorm2d, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum

        # just dummy buffers, not used
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))

        # projection layer
        self.weight_proj = nn.Linear(num_w, num_features)
        self.bias_proj = nn.Linear(num_w, num_features)

    def forward(self, x, w):
        b, c = x.size(0), x.size(1)
        running_mean = self.running_mean.repeat(b)
        running_var = self.running_var.repeat(b)

        weight, bias = self.weight_proj(w).contiguous().view(-1) + 1, self.bias_proj(w).contiguous().view(-1)

        # Apply instance norm
        x_reshaped = x.contiguous().view(1, b * c, *x.size()[2:])

        out = F.batch_norm(
            x_reshaped, running_mean, running_var, weight, bias,
            True, self.momentum, self.eps)

        return out.view(b, c, *x.size()[2:])

    def __repr__(self):
        return self.__class__.__name__ + '(' + str(self.num_features) + ')'


class SpatialAdaptiveInstanceNorm2d(nn.Module):
    def __init__(self, num_features, num_w=512, eps=1e-5, momentum=0.1):
        super(SpatialAdaptiveInstanceNorm2d, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum

        # just dummy buffers, not used
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))

        # projection layer
        self.weight_proj = nn.Linear(num_w, num_features)
        self.bias_proj = nn.Linear(num_w, num_features)

    def forward(self, x, w, bbox):
        b, c, h, w = x.size()
        running_mean = self.running_mean.repeat(b)
        running_var = self.running_var.repeat(b)
        return x


class AdaptiveBatchNorm2d(nn.BatchNorm2d):
    def __init__(self, num_features, num_w=512, eps=1e-5, momentum=0.1, affine=False, track_running_stats=True):
        super(AdaptiveBatchNorm2d, self).__init__(
            num_features, eps, momentum, affine, track_running_stats
        )
        # projection layer
        self.weight_proj = nn.Linear(num_w, num_features)
        self.bias_proj = nn.Linear(num_w, num_features)

    def forward(self, x, w):
        self._check_input_dim(x)
        exponential_average_factor = 0.0
        if self.training and self.track_running_stats:
            self.num_batches_tracked += 1
            if self.momentum is None:  # use cumulative moving average
                exponential_average_factor = 1.0 / self.num_batches_tracked.item()
            else:  # use exponential moving average
                exponential_average_factor = self.momentum

        output = F.batch_norm(x, self.running_mean, self.running_var,
                              self.weight, self.bias,
                              self.training or not self.track_running_stats,
                              exponential_average_factor, self.eps)

        size = output.size()
        weight, bias = self.weight_proj(w) + 1, self.bias_proj(w)
        weight = weight.unsqueeze(-1).unsqueeze(-1).expand(size)
        bias = bias.unsqueeze(-1).unsqueeze(-1).expand(size)

        return weight * output + bias

    def __repr__(self):
        return self.__class__.__name__ + '(' + str(self.num_features) + ')'


class SpatialAdaptiveBatchNorm2d(nn.BatchNorm2d):
    def __init__(self, num_features, num_w=512, eps=1e-5, momentum=0.1, affine=False,
                 track_running_stats=True):
        super(SpatialAdaptiveBatchNorm2d, self).__init__(
            num_features, eps, momentum, affine, track_running_stats
        )
        # projection layer
        self.weight_proj = nn.Linear(num_w, num_features)
        self.bias_proj = nn.Linear(num_w, num_features)

    def forward(self, x, vector, bbox):
        """
        :param x: input feature map (b, c, h, w)
        :param vector: latent vector (b*o, dim_w)
        :param bbox: bbox map (b, o, h, w)
        :return:
        """
        self._check_input_dim(x)
        exponential_average_factor = 0.0
        if self.training and self.track_running_stats:
            self.num_batches_tracked += 1
            if self.momentum is None:  # use cumulative moving average
                exponential_average_factor = 1.0 / self.num_batches_tracked.item()
            else:  # use exponential moving average
                exponential_average_factor = self.momentum

        output = F.batch_norm(x, self.running_mean, self.running_var,
                              self.weight, self.bias,
                              self.training or not self.track_running_stats,
                              exponential_average_factor, self.eps)

        b, o, _, _ = bbox.size()
        _, _, h, w = x.size()
        bbox = F.interpolate(bbox, size=(h, w), mode='bilinear')
        # calculate weight and bias
        weight, bias = self.weight_proj(vector), self.bias_proj(vector)

        weight, bias = weight.view(b, o, -1), bias.view(b, o, -1)

        weight = torch.sum(bbox.unsqueeze(2) * weight.unsqueeze(-1).unsqueeze(-1), dim=1, keepdim=False) / \
                 (torch.sum(bbox.unsqueeze(2), dim=1, keepdim=False) + 1e-6) + 1
        bias = torch.sum(bbox.unsqueeze(2) * bias.unsqueeze(-1).unsqueeze(-1), dim=1, keepdim=False) / \
               (torch.sum(bbox.unsqueeze(2), dim=1, keepdim=False) + 1e-6)
        return weight * output + bias

    def __repr__(self):
        return self.__class__.__name__ + '(' + str(self.num_features) + ')'


from .sync_batchnorm import SynchronizedBatchNorm2d


class SpatialAdaptiveSynBatchNorm2d(nn.Module):
    def __init__(self, num_features, num_w=512, batchnorm_func=SynchronizedBatchNorm2d, eps=1e-5, momentum=0.1, affine=False,
                 track_running_stats=True):
        super(SpatialAdaptiveSynBatchNorm2d, self).__init__()
        # projection layer
        self.num_features = num_features
        self.weight_proj = nn.utils.spectral_norm(nn.Linear(num_w, num_features))
        self.bias_proj = nn.utils.spectral_norm(nn.Linear(num_w, num_features))
        self.batch_norm2d = batchnorm_func(num_features, eps=eps, momentum=momentum,
                                           affine=affine)

    def forward(self, x, vector, bbox):
        """
        :param x: input feature map (b, c, h, w)
        :param vector: latent vector (b*o, dim_w)
        :param bbox: bbox map (b, o, h, w)
        :return:
        """

        # bbox is the resulting mask from the previous stage, M
        # the combination of the mask predicred by the mask regressor
        # and the clipped mask predicted by the previous ResBlock+ conv ToMask
        # not the bboxes coordinates

        # standard batch norm synchronized across devices to normalize features
        output = self.batch_norm2d(x)

        b, o, bh, bw = bbox.size()
        _, _, h, w = x.size()

        # adapt the mask to have the same size as the input features
        if bh != h or bw != w:
            bbox = F.interpolate(bbox, size=(h, w), mode='bilinear')
        
        # projection matrices learned from the label + style vector, A in the paper
        # calculate weight and bias, transformation parameters, Tau in the paper
        weight, bias = self.weight_proj(vector), self.bias_proj(vector)

        # resize weight and bias
        # (batch, num_o, num_features), (batch, num_o, num_features)
        weight, bias = weight.view(b, o, -1), bias.view(b, o, -1)

        # equation 8
        # M = bbox, weight = gamma (in Tau)
        # bbox.unsqueeze(2) * weight.unsqueeze(-1).unsqueeze(-1) has dim (batch, num_o, num_features, h, w)
        # sum over the objects, deleting the dimension afterwards
        # (batch, num_features, h, w)
        weight = torch.sum(bbox.unsqueeze(2) * weight.unsqueeze(-1).unsqueeze(-1), dim=1, keepdim=False) / \
                 (torch.sum(bbox.unsqueeze(2), dim=1, keepdim=False) + 1e-6) + 1
        # equation 9
        # M = bbox, bias = beta (in Tau)
        # bbox.unsqueeze(2) * bias.unsqueeze(-1).unsqueeze(-1) has dim (batch, num_o, num_features, h, w)
        # sum over the objects, deleting the dimension afterwards
        # (batch, num_features, h, w)
        bias = torch.sum(bbox.unsqueeze(2) * bias.unsqueeze(-1).unsqueeze(-1), dim=1, keepdim=False) / \
               (torch.sum(bbox.unsqueeze(2), dim=1, keepdim=False) + 1e-6)
        
        # transform the features
        return weight * output + bias

    def __repr__(self):
        return self.__class__.__name__ + '(' + str(self.num_features) + ')'
