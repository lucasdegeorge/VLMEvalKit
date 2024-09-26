import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import functional as F
import torchvision.transforms as transforms
from einops.layers.torch import Rearrange
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from llm.architecture.base import Attention
from vlm.config_zoo import VisionArgs, vision_configs

## Not finished yet !!


class ViTImageEncoder(nn.Module):
    def __init__(self, config: VisionArgs) -> None:
        super().__init__()
        self.config = config
        num_patches = (config.img_size // config.patch_size) * (
            config.img_size // config.patch_size
        )
        self.patch_embed = ConvPatchEmbedding(config)
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches + 1, config.embed_dim))
        self.layers = nn.ModuleList(
            ViTEncoderBlock(config) for _ in range(config.n_layers)
        )
        self.post_layer_norm = nn.LayerNorm(config.embed_dim)
        self.multi_modal_projector = MultiModalProjector(config)

    @classmethod
    def from_name(cls, name: str):
        return cls(VisionArgs.from_name(name))

    def forward(self, x: Tensor) -> Tensor:
        x = self.patch_embed(x)
        x = x + self.pos_embed
        for layer in self.layers:
            x = layer(x)
        x = self.post_layer_norm(x)
        x = self.multi_modal_projector(x)
        return x


class ViTEncoderBlock(nn.Module):
    def __init__(self, config: VisionArgs) -> None:
        super().__init__()
        self.config = config

        self.layer_norm1 = nn.LayerNorm(config.embed_dim)
        self.layer_norm2 = nn.LayerNorm(config.embed_dim)
        self.attention = Attention(config)
        self.feed_forward = FeedForwardForViT(config)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.attention(self.layer_norm1(x))
        x = x + self.feed_forward(self.layer_norm2(x))
        return x


class FeedForwardForViT(nn.Module):
    def __init__(self, config: VisionArgs) -> None:
        super().__init__()
        self.config = config
        self.w1 = nn.Linear(config.dim, config.intermediate_size)
        self.w2 = nn.Linear(config.intermediate_size, config.dim)

    def forward(self, x: Tensor) -> Tensor:
        x = self.w1(x)
        x = F.gelu(x)
        x = self.w2(x)
        return x


class MultiModalProjector(nn.Module):
    def __init__(self, config: VisionArgs) -> None:
        super().__init__()
        self.config = config
        self.linear = nn.Linear(config.embed_dim, config.out_channels)

    def forward(self, x: Tensor) -> Tensor:
        return self.linear(x)


class LinearPatchEmbedding(nn.Module):
    def __init__(self, config: VisionArgs) -> None:
        super().__init__()
        self.config = config
        self.patch_embed = nn.Sequential(
            Rearrange(
                "b c (h p1) (w p2) -> b (h w) (p1 p2 c)",
                p1=config.patch_size,
                p2=config.patch_size,
            ),
            nn.LayerNorm(config.patch_size),
            nn.Linear(
                config.patch_size * config.patch_size * config.in_channels,
                config.embed_dim,
            ),
            nn.LayerNorm(config.patch_size),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.patch_embed(x)


class ConvPatchEmbedding(nn.Module):
    def __init__(self, config: VisionArgs) -> None:
        super().__init__()
        self.config = config
        self.patch_embed = nn.Sequential(
            nn.Conv2d(
                config.in_channels,
                config.embed_dim,
                kernel_size=config.patch_size,
                stride=config.patch_size,
            ),
            Rearrange("b c h w -> b (h w) c"),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.patch_embed(x)


if __name__ == "__main__":
    model = ViTImageEncoder.from_name("test")
    path = ""
    checkpoint = torch.load(str(path), mmap=True, weights_only=True)
    model.load_state_dict(checkpoint, assign=True)
