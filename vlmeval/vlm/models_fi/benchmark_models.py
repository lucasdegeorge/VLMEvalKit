import torch
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from ..base import BaseModel
from models_fi.vlm.model import VLMBase


class BenchmarkModel(BaseModel):
    INTERLEAVE = False

    def __init__(self, model_name: str, compile: bool):
        super().__init__()
        self.vlm = VLMBase(
            model_name=model_name,
            compile=compile,
            quant="none",
            dtype=torch.bfloat16,
            device=torch.device("cuda"),
            # kv_cache_dtype=torch.float8_e5m2
        )

    def generate_inner(self, message, dataset=None):
        prompt, images = '', []
        for msg in message:
            if msg['type'] == 'text':
                prompt += msg['value']
            if msg['type'] == 'image':
                images.append(msg['value'])
        if len(images) == 1:
            output = self.vlm.generate(
                prompt=prompt, 
                image=images[0],
                max_new_tokens=512,
                do_decode=True,
                stop_first_eos=True,
                only_new_tokens=True,
                only_first_image_features=True,
                )
        if len(images) >= 2:
            raise NotImplementedError("Multiple images are not supported yet.")
        return output