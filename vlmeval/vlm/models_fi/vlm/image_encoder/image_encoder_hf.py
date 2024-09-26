import torch
from PIL import Image
from transformers import (
    PaliGemmaForConditionalGeneration,
    AutoProcessor,
    LlavaNextProcessor,
    LlavaNextForConditionalGeneration,
)
import torch.nn as nn
import numpy as np
import torchvision.transforms as transforms
from typing import Union, Tuple
import math
import os
import sys
from pathlib import Path
from trycast import isassignable

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from vlm.image_encoder.llava_utils import (
    image_size_to_num_patches,
    pack_image_features,
    merge_input_ids_with_image_features,
)
from vlm.config_zoo import VisionArgs
from llm.chat import Dialogue, DialogueForLLaVA

jean_zay = False
if jean_zay:
    checkpoints_path = Path("/gpfsdswork/projects/rech/fbe/uaa31dq/checkpoints/")
else:
    checkpoints_path = ""


class HFImageEncoder:
    def __init__(
        self,
        config: VisionArgs,
        do_delete_model: bool = False,
        device: Union[str, torch.device] = "cuda",
        dtype: torch.dtype = torch.float16,
    ) -> None:
        self.config = config
        self.device = device
        self.dtype = dtype
        assert (
            self.config.hf_model_id is not None
        ), "hf_model_id must be provided for HFImageEncoder"
        self.model, self.processor = self.load_model()
        self.vision_model = self.model.vision_tower
        self.projector = self.model.multi_modal_projector
        if do_delete_model:
            self.delete_model()
        self.max_new_tokens = 100

    @classmethod
    def from_name(cls, name: str, **args):
        return cls(VisionArgs.from_name(name), **args)

    def load_model(self) -> None:
        raise NotImplementedError("This method must be implemented in the child class")

    def prepare_prompt(self, prompt: str) -> str:
        raise NotImplementedError("This method must be implemented in the child class")

    def preprocess_prompt_image(
        self, prompt: str, image: Union[str, Image.Image, torch.Tensor]
    ) -> torch.Tensor:
        if isinstance(image, str):
            raw_image = Image.open(image)
        elif isinstance(image, Image.Image):
            raw_image = image
        elif isinstance(image, torch.Tensor):
            ## TODO: Add further checks on the tensor shape
            raw_image = transforms.ToPILImage()(image)
        else:
            raise ValueError("Image must be a str; a torch.Tensor or a PIL image")
        prompt = self.prepare_prompt(prompt)
        inputs = self.processor(prompt, raw_image, return_tensors="pt").to(self.device)
        return inputs

    def batch_preprocess_prompt_image(self, prompts: list, images: list) -> dict:
        raw_images, raw_prompts = [], []
        if images:
            for image in images:
                if isinstance(image, str):
                    raw_images.append(Image.open(image))
                elif isinstance(image, Image.Image):
                    raw_images.append(image)
                elif isinstance(image, torch.Tensor):
                    raw_images.append(transforms.ToPILImage()(image))
        else:
            raw_images = None
        assert isinstance(prompts, list) and all(isinstance(s, str) for s in prompts)
        raw_prompts = self.batch_prepare_prompt(prompts)
        inputs = self.processor(
            raw_prompts, raw_images, return_tensors="pt", padding=True
        )
        return inputs.to(self.device)

    def batch_prepare_prompt(self, prompts: list) -> list:
        raise NotImplementedError("This method must be implemented in the child class")

    def infer(self, prompt: str, image: str) -> str:
        raw_image = Image.open(image)
        prompt = self.prepare_prompt(prompt)
        inputs = self.processor(prompt, raw_image, return_tensors="pt").to(
            self.model.device
        )
        outputs = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens)
        return self.processor.decode(outputs[0], skip_special_tokens=True)

    def get_image_embeddings(
        self,
        image: Union[str, Image.Image, torch.Tensor],
        prompt: Union[str, torch.Tensor, Dialogue] = None,
    ) -> torch.Tensor:
        raise NotImplementedError("This method must be implemented in the child class")

    def merge_text_image_embeddings(
        self,
        text_embeddings: torch.Tensor,
        image: Union[str, Image.Image, torch.Tensor],
    ) -> torch.Tensor:
        """Image embdeddings are computed using the get_image_embeddings method within this method."""
        raise NotImplementedError("This method must be implemented in the child class")

    def delete_model(self):
        del self.model
        torch.cuda.empty_cache()


class HFPaliGemmaImageEncoder(HFImageEncoder):
    def __init__(
        self,
        config: VisionArgs = VisionArgs.from_name("hf-paligemma-3b-mix-224"),
        do_delete_model: bool = False,
        device: Union[str, torch.device] = "cuda",
        dtype: torch.dtype = torch.float16,
    ) -> None:
        super().__init__(config, do_delete_model, device, dtype)

    @classmethod
    def from_name(cls, name: str = "hf-paligemma-3b-mix-224", **args):
        return cls(VisionArgs.from_name(name), **args)

    def load_model(self):
        model = (
            PaliGemmaForConditionalGeneration.from_pretrained(
                self.config.hf_model_id, torch_dtype=self.dtype
            )
            .to(self.device)
            .eval()
        )
        processor = AutoProcessor.from_pretrained(
            self.config.hf_model_id,
            do_resize=True,
            resample=True,
            do_rescale=True,
            do_normalize=True,
        )
        return model, processor

    def prepare_prompt(self, prompt: str) -> str:
        return prompt

    def get_image_embeddings(
        self,
        image: Union[str, Image.Image, torch.Tensor],
        prompt: Union[str, torch.Tensor, Dialogue] = "",
    ) -> torch.Tensor:
        if isinstance(image, str):
            raw_image = Image.open(image)
        elif isinstance(image, Image.Image):
            raw_image = image
        elif isinstance(image, torch.Tensor):
            ## TODO: Add further checks on the tensor shape
            raw_image = transforms.ToPILImage()(image)
        else:
            raise ValueError("Image must be a str; a torch.Tensor or a PIL image")
        prompt = prompt if prompt else ""
        prompt = self.prepare_prompt(prompt)
        inputs = self.processor(prompt, raw_image, return_tensors="pt").to(self.device)
        vision_encoded = self.vision_model(inputs.pixel_values.to(dtype=self.dtype))
        outputs = self.projector(vision_encoded.last_hidden_state)
        return outputs

    def merge_text_image_embeddings(
        self,
        image: Union[str, Image.Image, torch.Tensor],
        text_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        """Image embdeddings are computed using the get_image_embeddings method within this method."""
        image_embeddings = self.get_image_embeddings(image, prompt="")
        image_embeddings /= self.config.embed_dim**0.5
        ti_embeddings = torch.cat(
            [image_embeddings[0], text_embeddings], dim=0
        ).unsqueeze(0)
        return ti_embeddings, image_embeddings


class HFLlavaNextImageEncoder(HFImageEncoder):
    def __init__(
        self,
        config: VisionArgs = VisionArgs.from_name("llama3-llava-next-8b-hf"),
        do_delete_model: bool = False,
        device: Union[str, torch.device] = "cuda",
        dtype: torch.dtype = torch.float16,
    ) -> None:
        super().__init__(config, do_delete_model, device, dtype)
        embed_std = 1 / math.sqrt(config.intermediate_size)
        self.image_newline = nn.Parameter(
            torch.randn(config.intermediate_size) * embed_std
        )
        # image_newline = torch.tensor(np.load('hf_image_newline.npy'))
        # print("image_newline", image_newline.shape, image_newline)
        # self.image_newline = nn.Parameter(image_newline)

    @classmethod
    def from_name(cls, name: str = "llama3-llava-next-8b-hf", **args):
        return cls(VisionArgs.from_name(name), **args)

    def load_model(self) -> None:
        model_path = checkpoints_path / Path(self.config.hf_model_id)
        model = (
            LlavaNextForConditionalGeneration.from_pretrained(
                model_path, torch_dtype=self.dtype, local_files_only=True
            )
            .to(self.device)
            .eval()
        )
        processor = LlavaNextProcessor.from_pretrained(
            model_path, local_files_only=True
        )
        return model, processor

    def prepare_prompt(self, prompt: str) -> str:
        if isassignable(prompt, DialogueForLLaVA):
            return self.processor.apply_chat_template(
                prompt, add_generation_prompt=True
            )
        else:
            conversation = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image"},
                    ],
                },
            ]
            prompt = self.processor.apply_chat_template(
                conversation, add_generation_prompt=True
            )
            return prompt

    def batch_prepare_prompt(self, prompts: list) -> list:
        return [f"USER: <image>\n{prompt} \nASSISTANT:" for prompt in prompts]

    def get_image_embeddings(
        self, inputs: dict
    ) -> Tuple[torch.Tensor, torch.Tensor, dict]:
        """Adapted from https://github.com/huggingface/transformers/blob/v4.44.0/src/transformers/models/llava_next/modeling_llava_next.py#L695"""

        image_num_patches = [
            image_size_to_num_patches(
                image_size=imsize,
                grid_pinpoints=self.config.image_grid_pinpoints,
                patch_size=self.config.img_size,
            )
            for imsize in inputs["image_sizes"]
        ]

        if inputs["pixel_values"].dim() == 5:
            # stacking when input is (batch_size, num_patches, num_channels, height, width)
            _pixel_values_list = [
                pix_val[:num_patch]
                for pix_val, num_patch in zip(inputs["pixel_values"], image_num_patches)
            ]
            inputs["pixel_values"] = torch.cat(_pixel_values_list, dim=0)
        elif inputs["pixel_values"].dim() != 4:
            raise ValueError(
                f"pixel_values of shape {inputs['pixel_values'].shape}, expect to be of 4 or 5 dimensions"
            )

        image_features = self.vision_model(
            inputs["pixel_values"], output_hidden_states=True
        )

        selected_image_feature = image_features.hidden_states[
            self.config.vision_features_layer
        ][:, 1:]

        image_features = self.projector(selected_image_feature)
        image_features = torch.split(image_features, image_num_patches, dim=0)
        image_features, feature_lens, base_image_features_size = pack_image_features(
            image_features,
            inputs["image_sizes"],
            image_newline=self.image_newline,
            image_size=self.config.img_size,
            patch_size=self.config.patch_size,
            image_grid_pinpoints=self.config.image_grid_pinpoints,
        )
        return image_features, feature_lens, inputs, base_image_features_size

    def merge_text_image_embeddings(
        self,
        inputs: dict,
        text_embeddings: torch.Tensor,
        only_first_image_features: bool = False,
    ) -> torch.Tensor:
        """inputs is and must be the output of the preprocess_prompt_image method"""

        if text_embeddings.dim() == 2:
            text_embeddings = text_embeddings.unsqueeze(0)

        image_features, feature_lens, inputs, base_image_features_size = (
            self.get_image_embeddings(inputs)
        )

        batch_size = inputs["input_ids"].size(0)
        ## Modified: image_features is now a list containing the bs features
        if only_first_image_features:
            image_features = [
                feature[:base_image_features_size] for feature in image_features
            ]
            feature_lens = torch.full((batch_size,), base_image_features_size)
        image_features = torch.cat(image_features, dim=0)

        final_embedding, final_input_ids = merge_input_ids_with_image_features(
            image_features,
            feature_lens,
            text_embeddings,
            inputs["input_ids"],
            inputs["attention_mask"],
            position_ids=None,
            image_token_index=self.config.image_token_index,
            pad_token_id=self.processor.tokenizer.pad_token_id,
        )
        return final_embedding, final_input_ids, image_features


if __name__ == "__main__":
    torch.manual_seed(45)
    model = HFLlavaNextImageEncoder()
    prompt = "Captions this image please"
    image = "/home/lucas/Documents/GitHub/fast_inference/tennis_player.jpeg"
    output = model.infer(prompt, image)
    print(output)
    # image = Image.open(image)
    # embs = model.get_image_embeddings(image)
    # print(embs.shape, embs)
