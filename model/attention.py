import torch
import torch.functional as F
from torch import nn, einsum
from einops import rearrange
from einops_exts.torch import EinopsToAndFrom


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


class Attention(nn.Module):
    def __init__(
        self,
        dim,
        *,
        dim_head=64,
        heads=8,
        context_dim=None,
        cosine_sim_attn=False
    ):
        super().__init__()
        self.scale = dim_head ** -0.5 if not cosine_sim_attn else 1.
        self.cosine_sim_attn = cosine_sim_attn
        self.cosine_sim_scale = 16 if cosine_sim_attn else 1

        self.heads = heads
        inner_dim = dim_head * heads

        self.norm = LayerNorm(dim)

        self.null_kv = nn.Parameter(torch.randn(2, dim_head))
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
        b, n, device = *x.shape[:2], x.device

        x = self.norm(x)
        q, k, v = (self.to_q(x), *self.to_kv(x).chunk(2, dim=-1))

        q = rearrange(q, 'b n (h d) -> b h n d', h=self.heads)
        q = q * self.scale

        # add null key / value for classifier free guidance in prior net

        # nk, nv = repeat_many(self.null_kv.unbind(dim=-2), 'd -> b 1 d', b=b)
        # k = torch.cat((nk, k), dim=-2)
        # v = torch.cat((nv, v), dim=-2)

        # add text conditioning, if present

        if exists(context):
            assert exists(self.to_context)
            ck, cv = self.to_context(context).chunk(2, dim=-1)
            k = torch.cat((ck, k), dim=-2)
            v = torch.cat((cv, v), dim=-2)

        # cosine sim attention

        if self.cosine_sim_attn:
            q, k = map(l2norm, (q, k))

        # calculate query / key similarities

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

        attn = sim.softmax(dim=-1)

        # aggregate values

        out = einsum('b h i j, b j d -> b h i d', attn, v)

        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class TransformerBlock(nn.Module):
    '''    Trasnformer block with multi-head attention    '''

    def __init__(
        self,
        dim,
        *,
        depth=1,
        heads=8,
        dim_head=32,
        context_dim=None,
        cosine_sim_attn=False
    ):
        '''
        Set parameters:
        - dim = c
        - context_dim = 1
    
        For:
            - x of size (b, c, h, w)
            - context of size (b, c, 1)
        '''
        super().__init__()
        self.layers = nn.ModuleList([])

        for _ in range(depth):
            self.layers.append(
                EinopsToAndFrom('b c h w', 'b (h w) c',
                                Attention(dim=dim, heads=heads, dim_head=dim_head,
                                          context_dim=context_dim, cosine_sim_attn=cosine_sim_attn))
            )

    def forward(self, x, context=None):
        for attn in self.layers:
            x = attn(x, context=context) + x
        return x
