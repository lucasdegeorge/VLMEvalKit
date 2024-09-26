from dataclasses import dataclass
from pathlib import Path
import os
import sys
from typing import Union

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from llm.architecture.base import LLMArgs
from llm.utils import find_multiple


@dataclass
class VisionArgs:
    img_size: Union[int, tuple] = None
    patch_size: Union[int, tuple] = None
    in_channels: int = None
    out_channels: int = None
    embed_dim: int = None  # projection_dim

    ## Args for text-image
    image_token_index: int = None
    n_image_tokens: int = None

    ## Args for FI models
    checkpoint_path: str = None
    block_size: int = None
    vocab_size: int = None
    n_layers: int = None
    n_heads: int = None
    dim: int = None  # hidden size
    intermediate_size: int = None
    n_local_heads: int = None
    head_dim: int = None
    biais: bool = None

    ## Args for HF models
    hf_model_id: str = None
    image_grid_pinpoints: list = None
    vision_features_layer: int = None

    def __post_init__(self):
        assert (
            self.checkpoint_path is not None or self.hf_model_id is not None
        ), "Either checkpoint_path or hf_model_id must be provided."
        if self.checkpoint_path is not None:
            self.checkpoint_path = Path(self.checkpoint_path)
            if self.n_local_heads == -1:
                self.n_local_heads = self.n_heads
            if self.intermediate_size is None:
                hidden_dim = 4 * self.dim
                n_hidden = int(2 * hidden_dim / 3)
                self.intermediate_size = find_multiple(n_hidden, 256)
            if self.head_dim is None:
                self.head_dim = self.dim // self.n_heads
        if self.hf_model_id is not None:
            if "llava" in self.hf_model_id:
                assert (
                    self.image_grid_pinpoints is not None
                ), "image_grid_pinpoints must be provided for HF llava-next models."
                assert (
                    self.vision_features_layer is not None
                ), "vision_features_layer must be provided for HF llava-next models."

    @classmethod
    def from_name(cls, name: str):
        if name in vision_configs:
            return cls(**vision_configs[name])
        else:
            raise ValueError(f"Unknown model name: {name}")


@dataclass
class VLMArgs:
    llm_args: LLMArgs
    vision_args: VisionArgs

    @classmethod
    def from_name(cls, name: str):
        if name in vlm_configs:
            return cls(**vlm_configs[name])
        else:
            raise ValueError(f"Unknown VLM model name: {name}")


vision_configs = {
    "llama3-llava-next-8b-hf": dict(
        image_token_index=128256,
        img_size=336,
        patch_size=14,
        intermediate_size=4096,
        n_heads=16,
        n_layers=24,
        embed_dim=728,
        hf_model_id="llava-hf/llama3-llava-next-8b-hf",
        image_grid_pinpoints=[
            [336, 672],
            [672, 336],
            [672, 672],
            [1008, 336],
            [336, 1008],
        ],
        vision_features_layer=-2,
    ),
    "hf-paligemma-3b-mix-224": dict(
        checkpoint_path="checkpoints/google/paligemma-3b-mix-224",
        hf_model_id="google/paligemma-3b-mix-224",
        block_size=8192,
        dim=1152,
        vocab_size=257216,
        n_layers=27,
        n_heads=16,
        n_local_heads=-1,
        intermediate_size=4304,
        patch_size=14,
        embed_dim=2048,
        image_token_index=257152,
        n_image_tokens=256,
    ),
    "hf-paligemma-3b-pt-896": dict(
        checkpoint_path="checkpoints/google/paligemma-3b-pt-896",
        hf_model_id="google/paligemma-3b-pt-896",
        block_size=8192,
        dim=1152,
        vocab_size=257216,
        n_layers=27,
        n_heads=16,
        n_local_heads=-1,
        intermediate_size=4304,
        patch_size=14,
        embed_dim=2048,
        image_token_index=257152,
        n_image_tokens=4096,
        img_size=896,
    ),
}


vlm_configs = {
    "llama3-llava-next-8b-hf": dict(
        llm_args=LLMArgs.from_name("llama3-llava-next-8b-hf"),
        vision_args=VisionArgs.from_name("llama3-llava-next-8b-hf"),
    ),
    "hf-paligemma-3b-mix-224": dict(
        llm_args=LLMArgs.from_name("paligemma-3b-mix-224"),
        vision_args=VisionArgs.from_name("hf-paligemma-3b-mix-224"),
    ),
    "hf-paligemma-3b-pt-896": dict(
        llm_args=LLMArgs.from_name("paligemma-3b-pt-896"),
        vision_args=VisionArgs.from_name("hf-paligemma-3b-pt-896"),
    ),
}
