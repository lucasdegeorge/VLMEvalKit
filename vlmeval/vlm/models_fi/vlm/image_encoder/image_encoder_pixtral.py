import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional
import os
import sys
from xformers.ops.fmha.attn_bias import BlockDiagonalMask

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from llm.architecture.base import RMSNorm, Transformer
from vlm.config_zoo import VLMArgs
from vlm.image_encoder.pixtral_utils import (
    precompute_freqs_cis_2d,
    position_meshgrid,
)


"""
class VisionEncoderArgs:  -> class VisionArgs
    hidden_size: int  -> dim
    num_channels: int  -> in_channels
    image_size: int  -> img_size
    patch_size: int
    intermediate_size: int
    num_hidden_layers: int  -> n_layers
    num_attention_heads: int  -> n_heads
    rope_theta: float  # for rope-2D  -> LLMArgs.rope_theta
    image_token_id: int
"""


class PixtralViT(nn.Module):
    """Adapted from https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/models/pixtral.py#L471"""

    def __init__(self, config: VLMArgs):
        super().__init__()
        self.config = config
        self.patch_conv = nn.Conv2d(
            in_channels=config.vision_args.in_channels,
            out_channels=config.vision_args.dim,  ## hidden_size
            kernel_size=config.vision_args.patch_size,
            stride=config.vision_args.patch_size,
            bias=False,
        )
        self.ln_pre = RMSNorm(config.vision_args.dim, eps=1e-5)
        self.transformer = Transformer(config.llm_args)

        head_dim = self.config.vision_args.dim // self.config.vision_args.n_heads
        assert head_dim % 2 == 0, "ROPE requires even head_dim"
        self._freqs_cis: Optional[torch.Tensor] = None

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    @property
    def dtype(self) -> torch.device:
        return next(self.parameters()).dtype

    @property
    def freqs_cis(self) -> torch.Tensor:
        if self._freqs_cis is None:
            self._freqs_cis = precompute_freqs_cis_2d(
                dim=self.config.vision_args.dim // self.config.vision_args.n_heads,
                height=self.config.vision_args.img_size
                // self.config.vision_args.patch_size,
                width=self.config.vision_args.img_size
                // self.config.vision_args.patch_size,
                theta=self.config.llm_args.rope_theta,
            )

        if self._freqs_cis.device != self.device:
            self._freqs_cis = self._freqs_cis.to(device=self.device)

        return self._freqs_cis

    def forward(
        self,
        images: List[torch.Tensor],
    ) -> torch.Tensor:
        """
        Args:
            images: list of N_img images of variable sizes,
                each of shape (C, H, W)
        Returns:
            image_features: tensor of token features for
                all tokens of all images of shape (N_toks, D)
        """
        patch_embeds_list = [
            self.patch_conv(img.unsqueeze(0).to(self.dtype)) for img in images
        ]

        patch_embeds = torch.cat(
            [p.flatten(2).permute(0, 2, 1) for p in patch_embeds_list], dim=1
        )
        patch_embeds = self.ln_pre(patch_embeds)

        positions = position_meshgrid(patch_embeds_list).to(self.device)
        freqs_cis = self.freqs_cis[positions[:, 0], positions[:, 1]]

        mask = BlockDiagonalMask.from_seqlens(
            [p.shape[-2] * p.shape[-1] for p in patch_embeds_list],
        )
        out = self.transformer(
            idx=None,
            input_pos=None,
            embeds=patch_embeds,
            left_pad_mask_pos=None,
            precomputed_freqs_cis=freqs_cis,
        )

        return out.squeeze(0)
