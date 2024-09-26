import torch
from torch import Tensor
import torch.nn as nn
from typing import Optional
from torch.nn import functional as F
from .base import (
    Transformer,
    FeedForward,
    RMSNorm,
    Attention,
    LLMArgs,
)


class TransformerForGemma(Transformer):
    def __init__(self, config):
        super().__init__(config)

        ## Lines with new classes for Gemma models
        self.layers = nn.ModuleList(
            TransformerBlockForGemma(config) for _ in range(config.n_layers)
        )
        self.norm = RMSNormForGemma(config.dim, eps=config.rms_norm_eps)

    def forward(
        self,
        idx: Tensor,
        input_pos: Optional[Tensor] = None,
        left_pad_mask_pos: Optional[Tensor] = None,
        fully_causal: bool = False,
    ) -> Tensor:
        assert self.freqs_cis is not None, "Caches must be initialized first"
        assert not (fully_causal and left_pad_mask_pos is not None), "Invalid mask"

        mask = self.causal_mask[None, None, input_pos]
        if left_pad_mask_pos is not None:
            pad_mask = torch.arange(mask.size(-1), device=mask.device).view(
                1, -1
            ) >= left_pad_mask_pos.view(-1, 1)
            mask = torch.logical_and(mask, pad_mask[:, None, None, :].contiguous())
            mask = torch.logical_or(mask, self.self_mask[None, None, input_pos])

        freqs_cis = self.freqs_cis[input_pos]
        x = self.tok_embeddings(idx)
        ## New line for Gemma models
        x = (self.config.dim**0.5) * x

        for i, layer in enumerate(self.layers):
            x = layer(x, input_pos, freqs_cis, mask)
        x = self.norm(x)
        logits = self.output(x)
        return logits


class TransformerBlockForGemma(nn.Module):
    def __init__(self, config: LLMArgs) -> None:
        super().__init__()
        self.attention = Attention(config)
        self.feed_forward = FeedForward(config)
        self.ffn_norm = RMSNormForGemma(
            config.dim, config.rms_norm_eps
        )  ## RMSNormForGemma instead of RMSNorm
        self.attention_norm = RMSNormForGemma(
            config.dim, config.rms_norm_eps
        )  ## RMSNormForGemma instead of RMSNorm

    def forward(
        self, x: Tensor, input_pos: Tensor, freqs_cis: Tensor, mask: Tensor
    ) -> Tensor:
        h = x + self.attention(self.attention_norm(x), freqs_cis, mask, input_pos)
        out = h + self.feed_forward(self.ffn_norm(h))
        return out


class RMSNormForGemma(RMSNorm):
    def __init__(self, dim, eps):
        super().__init__(dim, eps)

    def forward(self, x: Tensor) -> Tensor:
        output = self._norm(x.float()).type_as(x)
        ## New line for Gemma models: 1 + self.weight instead of self.weight
        return output * (1 + self.weight)
