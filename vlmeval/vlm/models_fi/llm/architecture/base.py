from dataclasses import dataclass
from typing import Optional, Union, Dict
from pathlib import Path

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import functional as F

from llm.utils import (
    activation_functions,
    find_multiple,
    precompute_freqs_cis,
    hf_rotary_emb,
    fi_rotary_emb,
    # vit_apply_rotary_emb,
)
from llm.config_zoo import transformer_configs


@dataclass
class LLMArgs:
    checkpoint_path: str = None
    block_size: int = 2048
    vocab_size: int = 32000
    n_layers: int = 32
    n_heads: int = 32
    dim: int = 4096
    intermediate_size: int = None
    n_local_heads: int = -1
    head_dim: int = None
    rms_norm_eps: float = 1e-5
    biais: bool = False
    rope_type: str = "fi"
    rope_base: float = 10000
    rope_theta: float = 500000.0
    max_position_embeddings: int = 8192
    activation: str = "silu"

    def __post_init__(self):
        assert self.checkpoint_path is not None, "checkpoint_path must be provided"
        assert self.activation in activation_functions, "Invalid activation function"
        self.checkpoint_path = Path(self.checkpoint_path)
        if self.n_local_heads == -1:
            self.n_local_heads = self.n_heads
        if self.intermediate_size is None:
            hidden_dim = 4 * self.dim
            n_hidden = int(2 * hidden_dim / 3)
            self.intermediate_size = find_multiple(n_hidden, 256)
        if self.head_dim is None:
            self.head_dim = self.dim // self.n_heads

    @classmethod
    def from_name(cls, name: str):
        if name in transformer_configs:
            return cls(**transformer_configs[name])
        else:
            raise ValueError(f"Unknown model name: {name}")


class Transformer(nn.Module):
    def __init__(self, config: LLMArgs) -> None:
        super().__init__()
        self.config = config

        self.tok_embeddings = nn.Embedding(config.vocab_size, config.dim)
        self.layers = nn.ModuleList(
            TransformerBlock(config) for _ in range(config.n_layers)
        )
        self.norm = RMSNorm(config.dim, eps=config.rms_norm_eps)
        self.output = nn.Linear(config.dim, config.vocab_size, bias=config.biais)

        self.freqs_cis: Optional[Tensor] = None
        self.mask_cache: Optional[Tensor] = None
        self.max_batch_size = -1
        self.max_seq_length = -1
        self.vocab_parallel = False

    def get_tok_embeddings(self):
        return self.tok_embeddings

    def setup_caches(
        self,
        max_batch_size,
        max_seq_length,
        prompt_size: int,
        device: torch.device,
        kv_dtype: torch.dtype = None,
        precomputed_freqs_cis: Optional[Tensor] = None,
    ):
        if (
            self.max_seq_length >= max_seq_length
            and self.max_batch_size >= max_batch_size
        ):
            return
        max_seq_length = find_multiple(max_seq_length, 8)
        self.max_seq_length = max_seq_length
        self.max_batch_size = max_batch_size
        dtype = self.output.weight.dtype
        # For quantized layers, dtype is encoded in scales
        if hasattr(self.output, "scales"):
            dtype = self.output.scales.dtype
        elif hasattr(self.output, "scales_and_zeros"):
            dtype = self.output.scales_and_zeros.dtype

        kv_dtype = dtype if kv_dtype is None else kv_dtype
        for b in self.layers:
            b.attention.kv_cache = KVCache(
                max_batch_size,
                max_seq_length,
                self.config.n_local_heads,
                self.config.head_dim,
                kv_dtype,
            )
        if precomputed_freqs_cis is not None:
            self.freqs_cis = precomputed_freqs_cis
        else:
            self.freqs_cis = precompute_freqs_cis(
                self.config.block_size,
                self.config.head_dim,
                self.config.rope_base,
                dtype,
                device,
            )
        self.causal_mask = torch.tril(
            torch.ones(
                self.max_seq_length,
                self.max_seq_length,
                dtype=torch.bool,
                device=device,
            )
        )
        # self.causal_mask[:prompt_size, :prompt_size] = torch.ones_like(
        #     self.causal_mask[:prompt_size, :prompt_size]
        # ).to(device=device, dtype=torch.bool)

        self.self_mask = torch.eye(self.max_seq_length, dtype=torch.bool, device=device)

    def forward(
        self,
        idx: Tensor,
        input_pos: Optional[Tensor] = None,
        embeds: Optional[Tensor] = None,
        left_pad_mask_pos: Optional[Tensor] = None,
        precomputed_freqs_cis: Optional[Tensor] = None,
        precomputed_mask: Optional[Tensor] = None,
        fully_causal: bool = False,
    ) -> Tensor:
        assert (
            self.freqs_cis is not None or precomputed_freqs_cis is not None
        ), "Caches must be initialized first"
        assert not (fully_causal and left_pad_mask_pos is not None), "Invalid mask"

        if not precomputed_mask:
            mask = self.causal_mask[None, None, input_pos]
            if left_pad_mask_pos is not None:
                pad_mask = torch.arange(mask.size(-1), device=mask.device).view(
                    1, -1
                ) >= left_pad_mask_pos.view(-1, 1)
                mask = torch.logical_and(mask, pad_mask[:, None, None, :].contiguous())
                mask = torch.logical_or(mask, self.self_mask[None, None, input_pos])
        else:
            mask = precomputed_mask

        if precomputed_freqs_cis is not None:
            freqs_cis = precomputed_freqs_cis
        else:
            freqs_cis = self.freqs_cis[input_pos]

        if embeds is not None:
            x = embeds
        else:
            x = self.tok_embeddings(idx)

        for i, layer in enumerate(self.layers):
            x = layer(x, input_pos, freqs_cis, mask)
        x = self.norm(x)
        logits = self.output(x)
        return logits

    @classmethod
    def from_name(cls, name: str):
        return cls(LLMArgs.from_name(name))


class KVCache(nn.Module):
    def __init__(self, max_batch_size, max_seq_length, n_heads, head_dim, dtype):
        super().__init__()
        self.dtype = dtype
        cache_shape = (max_batch_size, n_heads, max_seq_length, head_dim)
        self.register_buffer("k_cache", torch.zeros(cache_shape, dtype=dtype))
        self.register_buffer("v_cache", torch.zeros(cache_shape, dtype=dtype))

    def update(self, input_pos, k_val, v_val):
        # input_pos: [S], k_val: [B, H, S, D]
        assert input_pos.shape[0] == k_val.shape[2]

        k_out = self.k_cache
        v_out = self.v_cache
        k_out[: k_val.size(0), :, input_pos] = k_val.to(self.dtype)
        v_out[: k_val.size(0), :, input_pos] = v_val.to(self.dtype)

        return k_out[: k_val.size(0)].to(k_val.dtype), v_out[: k_val.size(0)].to(
            k_val.dtype
        )


class TransformerBlock(nn.Module):
    def __init__(self, config: LLMArgs) -> None:
        super().__init__()
        self.attention = Attention(config)
        self.feed_forward = FeedForward(config)
        self.ffn_norm = RMSNorm(config.dim, config.rms_norm_eps)
        self.attention_norm = RMSNorm(config.dim, config.rms_norm_eps)

    def forward(
        self,
        x: Tensor,
        input_pos: Tensor,
        freqs_cis: Tensor,
        mask: Union[Tensor, str],
        fully_causal: bool = False,
    ) -> Tensor:

        ## Lines for VLM models
        # if mask.shape[2] > 1:
        #     inp_size = mask.shape[2]
        #     mask[:, :, :inp_size, :inp_size] = torch.ones_like(
        #         mask[:, :, :inp_size, :inp_size]
        #     )

        h = x + self.attention(
            self.attention_norm(x), freqs_cis, mask, input_pos, fully_causal
        )
        out = h + self.feed_forward(self.ffn_norm(h))
        return out


class Attention(nn.Module):
    def __init__(self, config: LLMArgs):
        super().__init__()
        assert config.dim % config.n_heads == 0
        self.config = config

        total_head_dim = (config.n_heads + 2 * config.n_local_heads) * config.head_dim
        # key, query, value projections for all heads, but in a batch
        self.wqkv = nn.Linear(config.dim, total_head_dim, bias=config.biais)
        self.wo = nn.Linear(
            config.n_heads * config.head_dim, config.dim, bias=config.biais
        )
        self.kv_cache = None

        self.n_heads = config.n_heads
        self.head_dim = config.head_dim
        self.n_local_heads = config.n_local_heads
        self.dim = config.dim
        self._register_load_state_dict_pre_hook(self.load_hook)

        self.rope_type = config.rope_type
        self.rope = None

    def load_hook(self, state_dict, prefix, *args):
        if prefix + "wq.weight" in state_dict:
            wq = state_dict.pop(prefix + "wq.weight")
            wk = state_dict.pop(prefix + "wk.weight")
            wv = state_dict.pop(prefix + "wv.weight")
            state_dict[prefix + "wqkv.weight"] = torch.cat([wq, wk, wv])
        if prefix + "wq.biais" in state_dict:
            bq = state_dict.pop(prefix + "wq.biais")
            bk = state_dict.pop(prefix + "wk.biais")
            bv = state_dict.pop(prefix + "wv.biais")
            state_dict[prefix + "wqkv.biais"] = torch.cat([bq, bk, bv])

    def apply_rotary_emb(self, q, k, v, freqs_cis, input_pos: None):
        if self.rope_type == "fi" or self.rope_type == ("fi",):
            q = fi_rotary_emb(q, freqs_cis)
            k = fi_rotary_emb(k, freqs_cis)
            q, k, v = map(lambda x: x.transpose(1, 2), (q, k, v))
            return q, k, v
        elif self.rope_type == "hf" or self.rope_type == ("hf",):
            if self.rope is None:
                self.rope = HuggingFaceRotaryEmbedding(self.config)
            q, k, v = map(lambda x: x.transpose(1, 2), (q, k, v))
            assert self.rope is not None
            cos, sin = self.rope(v, input_pos)
            q = hf_rotary_emb(q, cos, sin)
            k = hf_rotary_emb(k, cos, sin)
            return q, k, v
        # elif self.rope_type == "vit" or self.rope_type == ("vit",):
        #     q = vit_apply_rotary_emb(q, freqs_cis)
        #     k = vit_apply_rotary_emb(k, freqs_cis)
        #     q, k, v = map(lambda x: x.transpose(1, 2), (q, k, v))
        #     return q, k, v
        elif self.rope_type == "none" or self.rope_type == ("none",):
            return q, k, v
        else:
            raise ValueError(f"Invalid RoPE type: {self.rope_type}")

    def forward(
        self,
        x: Tensor,
        freqs_cis: Tensor,
        mask: Union[Tensor, str],
        input_pos: Optional[Tensor] = None,
        fully_causal: bool = False,
    ) -> Tensor:
        batch_size, seqlen, _ = x.shape

        kv_size = self.n_local_heads * self.head_dim
        q, k, v = self.wqkv(x).split(
            [self.n_heads * self.head_dim, kv_size, kv_size], dim=-1
        )

        q = q.view(batch_size, seqlen, self.n_heads, self.head_dim)
        k = k.view(batch_size, seqlen, self.n_local_heads, self.head_dim)
        v = v.view(batch_size, seqlen, self.n_local_heads, self.head_dim)

        q, k, v = self.apply_rotary_emb(q, k, v, freqs_cis, input_pos)

        if self.kv_cache is not None and not fully_causal:
            k, v = self.kv_cache.update(input_pos, k, v)

        k = k.repeat_interleave(self.n_heads // self.n_local_heads, dim=1)
        v = v.repeat_interleave(self.n_heads // self.n_local_heads, dim=1)

        if q.size(2) == k.size(2) and fully_causal:
            with torch.backends.cuda.sdp_kernel(
                enable_flash=True, enable_math=False, enable_mem_efficient=False
            ):
                y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        else:
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=0.0)

        # y = (
        #     y.transpose(1, 2)
        #     .contiguous()
        #     .view(batch_size, seqlen, self.n_heads * self.head_dim)
        # )
        y = y.transpose(1, 2).reshape(batch_size, seqlen, self.n_heads * self.head_dim)
        y = self.wo(y)
        return y


class FeedForward(nn.Module):
    def __init__(self, config: LLMArgs) -> None:
        super().__init__()
        self.w1 = nn.Linear(config.dim, config.intermediate_size, bias=config.biais)
        self.w3 = nn.Linear(config.dim, config.intermediate_size, bias=config.biais)
        self.w2 = nn.Linear(config.intermediate_size, config.dim, bias=config.biais)
        self.activation = activation_functions[config.activation]

    def forward(self, x: Tensor) -> Tensor:
        return self.w2(self.activation(self.w1(x)) * self.w3(x))


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(torch.mean(x * x, dim=-1, keepdim=True) + self.eps)

    def forward(self, x: Tensor) -> Tensor:
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


class HuggingFaceRotaryEmbedding(nn.Module):
    def __init__(self, config: LLMArgs) -> None:
        super().__init__()
        dim = int(config.dim // config.n_heads)
        self.inv_freq = 1.0 / (
            config.rope_theta
            ** (torch.arange(0, dim, 2, dtype=torch.int64).float() / dim)
        )
        self.attention_scaling = 1.0

        # self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.original_inv_freq = self.inv_freq

    @torch.no_grad()
    def forward(self, x, position_ids):
        position_ids = position_ids.unsqueeze(0)
        self.inv_freq = self.inv_freq.to(device=x.device)

        # Core RoPE block
        inv_freq_expanded = (
            self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        )
        position_ids_expanded = position_ids[:, None, :].float()

        device_type = x.device.type
        device_type = (
            device_type
            if isinstance(device_type, str) and device_type != "mps"
            else "cpu"
        )
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (
                inv_freq_expanded.float() @ position_ids_expanded.float()
            ).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos()
            sin = emb.sin()

        cos = cos * self.attention_scaling
        sin = sin * self.attention_scaling

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)
