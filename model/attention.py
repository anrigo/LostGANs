import torch
import torch.functional as F
from torch import nn, einsum
from einops import rearrange


def exists(val):
    return val is not None


def l2norm(t):
    return F.normalize(t, dim=-1)


class LayerNorm(nn.Module):
    def __init__(self, dim, eps=1e-5, stable=False):
        super().__init__()
        self.eps = eps
        self.stable = stable
        self.g = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        if self.stable:
            x = x / x.amax(dim=-1, keepdim=True).detach()

        var = torch.var(x, dim=-1, unbiased=False, keepdim=True)
        mean = torch.mean(x, dim=-1, keepdim=True)
        return (x - mean) * (var + self.eps).rsqrt() * self.g


class ChanLayerNorm(nn.Module):
    def __init__(self, dim, eps=1e-5, stable=False):
        super().__init__()
        self.eps = eps
        self.stable = stable
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))

    def forward(self, x):
        if self.stable:
            x = x / x.amax(dim=1, keepdim=True).detach()

        var = torch.var(x, dim=1, unbiased=False, keepdim=True)
        mean = torch.mean(x, dim=1, keepdim=True)
        return (x - mean) / (var + self.eps).sqrt() * self.g


# in paper, it seems for self attention layers they did feedforwards with twice channel width
def ChanFeedForward(dim, mult=2):
    hidden_dim = int(dim * mult)
    return nn.Sequential(
        ChanLayerNorm(dim),
        nn.Conv2d(dim, hidden_dim, 1, bias=False),
        nn.GELU(),
        ChanLayerNorm(hidden_dim),
        nn.Conv2d(hidden_dim, dim, 1, bias=False)
    )


class Attention(nn.Module):
    def __init__(
        self,
        dim,
        *,
        dim_head=64,
        heads=8,
        context_dim=None,
        cosine_sim_attn=False,
        crossonly=False
    ):
        super().__init__()
        self.scale = dim_head ** -0.5 if not cosine_sim_attn else 1.
        self.cosine_sim_attn = cosine_sim_attn
        self.cosine_sim_scale = 16 if cosine_sim_attn else 1

        # default: self-attention + cross-attention
        # self-attention only: context_dim = None, crossonly = False
        # cross-attention only: crossonly = True, context_dim = m
        self.crossonly = crossonly

        self.heads = heads
        inner_dim = dim_head * heads

        self.norm = LayerNorm(dim)

        # self.null_kv = nn.Parameter(torch.randn(2, dim_head))
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim, dim_head * 2, bias=False)

        self.to_context = nn.Sequential(
            nn.LayerNorm(context_dim),
            nn.Linear(context_dim, dim_head * 2)) if exists(context_dim) else None

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim, bias=False),
            LayerNorm(dim)
        )

    def forward(self, x, context=None, mask=None, attn_bias=None):
        # n = h*w
        b, n, device = *x.shape[:2], x.device

        # (b, n, c)
        x = self.norm(x)
        # (b, n, dim), (b, n, dim_head), (b, n, dim_head)
        q, k, v = (self.to_q(x), *self.to_kv(x).chunk(2, dim=-1))
        # (b, heads, n, dim_head)
        q = rearrange(q, 'b n (h d) -> b h n d', h=self.heads)
        q = q * self.scale

        # add depth conditioning, if present

        if exists(context) and exists(self.to_context):
            # (b, num_o, dim_head)
            ck, cv = self.to_context(context).chunk(2, dim=-1)

            if not self.crossonly:
                # (b, n+num_o, dim_head)
                k = torch.cat((ck, k), dim=-2)
                v = torch.cat((cv, v), dim=-2)
            else:
                k, v = ck, cv

        # cosine sim attention

        if self.cosine_sim_attn:
            q, k = map(l2norm, (q, k))

        # calculate query / key similarities
        # (b, heads, n, n+num_o)
        sim = einsum('b h i d, b j d -> b h i j', q, k) * self.cosine_sim_scale

        # relative positional encoding (T5 style)

        if exists(attn_bias):
            sim = sim + attn_bias

        # masking

        max_neg_value = -torch.finfo(sim.dtype).max

        if exists(mask):
            mask = F.pad(mask, (1, 0), value=True)
            mask = rearrange(mask, 'b j -> b 1 1 j')
            sim = sim.masked_fill(~mask, max_neg_value)

        # attention
        # (b, heads, n, n+num_o)
        attn = sim.softmax(dim=-1)

        # aggregate values
        # (b, heads, n, dim_head)
        out = einsum('b h i j, b j d -> b h i d', attn, v)
        # (b, n, heads*dim_head)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out), attn


class EinopsToAndFromTuple(nn.Module):
    def __init__(self, from_einops, to_einops, fn):
        super().__init__()
        self.from_einops = from_einops
        self.to_einops = to_einops
        self.fn = fn

    def forward(self, x, **kwargs):
        shape = x.shape
        reconstitute_kwargs = dict(tuple(zip(self.from_einops.split(' '), shape)))
        x = rearrange(x, f'{self.from_einops} -> {self.to_einops}')
        x = self.fn(x, **kwargs)

        if isinstance(x, tuple):
            x, *rest = x

        x = rearrange(x, f'{self.to_einops} -> {self.from_einops}', **reconstitute_kwargs)
        return x, *rest


class TransformerBlock(nn.Module):
    '''    Trasnformer block with multi-head attention    '''

    def __init__(
        self,
        dim,
        *,
        depth=1,
        heads=8,
        dim_head=32,
        ff_mult=2,
        context_dim=None,
        cosine_sim_attn=False,
        crossonly=False
    ):
        '''
        Set parameters:
            - dim = c
            - context_dim = m
    
        For:
            - x of size (b, c, h, w)
            - context of size (b, c, m)
        
        default: self-attention + cross-attention
        self-attention only: context_dim = None, crossonly = False
        cross-attention only: crossonly = True, context_dim = m
        '''
        super().__init__()
        self.layers = nn.ModuleList([])

        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                EinopsToAndFromTuple('b c h w', 'b (h w) c',
                                Attention(dim=dim, heads=heads, dim_head=dim_head,
                                          context_dim=context_dim, cosine_sim_attn=cosine_sim_attn, crossonly=crossonly)),
                ChanFeedForward(dim=dim, mult=ff_mult)
            ]))

    def forward(self, x, context=None, return_attn=False):
        for attn, ff in self.layers:
            # x = attn(x, context=context) + x
            ft, attn_scores = attn(x, context=context)
            x = ft + x
            x = ff(x) + x
        if return_attn:
            h, w = x.shape[-2:]
            return x, attn_scores.view(*attn_scores.shape[:2], h, w, -1)
        return x
