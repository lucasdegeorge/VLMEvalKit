from trycast import isassignable
import torch
from typing import TypedDict, Sequence, Literal, List, Union, Optional
from PIL import Image
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from llm.model import LLMBase
from llm.chat import ChatFormat, Dialogue, DialogueForVQA, DialogueForLLaVA
from vlm.model import VLMBase, GemmaVLMBase
from llm.tokenizer import HuggingFaceWrapper


class Llama3ForVQA(LLMBase):
    def __init__(
        self,
        model_name: str = "Meta-Llama-3-8B-Instruct",
        compile: bool = True,
        quant: str = "int8",
        use_tp: bool = False,
        dtype: torch.dtype = torch.bfloat16,
        kv_cache_dtype: torch.dtype = None,
        device: Union[str, torch.device] = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        ),
    ):
        super().__init__(
            model_name, compile, quant, use_tp, dtype, kv_cache_dtype, device
        )
        assert "Llama-3" in model_name, "This class is only for Llama-3 models"

    def apply_chat_template(
        self, conversation: Dialogue, return_tensor: bool = True, **kwargs
    ) -> Union[str, torch.Tensor]:
        assert isassignable(
            conversation, Dialogue
        ), "conversation must be a Dialogue for Llama-3 model"
        assert isinstance(
            self.chat_template, ChatFormat
        ), "chat_template must be a ChatFormat for Llama-3 model"
        return self.chat_template.encode_dialogue_prompt(
            conversation, return_tensor=return_tensor
        )

    @property
    def chat_template(self):
        return ChatFormat(self.tokenizer)

    def dialogue_from_dialogue_vqa(self, dialogue: DialogueForVQA, is_last):
        """Version for LLM"""
        res = []
        for message in dialogue:
            if message["role"] == "initial":
                res.append({"role": "system", "content": message["content"]})
            elif message["role"] == "llm":
                res.append({"role": "assistant", "content": message["content"]})
            elif message["role"] == "vlm":
                if is_last:
                    res.append({"role": "user", "content": message["content"]})
                else:
                    res.append(
                        {
                            "role": "user",
                            "content": message["content"]
                            + " Ask me a new questions. Do not focus on the same object at each question. Only answer by a new question. Do not say anything else.",
                        }
                    )
        return res

    def get_question(self, dialogue: DialogueForVQA, is_last=False, **sampling_kwargs):
        prompt = self.dialogue_from_dialogue_vqa(dialogue, is_last)
        prompt = self.apply_chat_template(prompt)
        question = self.generate(
            prompt, max_new_tokens=200, do_decode=True, **sampling_kwargs
        )[0]
        question = (
            question.replace("\n", "")
            .split("<|end_header_id|>")[-1]
            .split("<|eot_id|>")[0]
        )
        dialogue.append({"role": "llm", "content": question})
        return dialogue


class LLaVAForVQA(VLMBase):
    def __init__(
        self,
        model_name: str = "llama3-llava-next-8b-hf",
        compile: bool = True,
        quant: str = "none",
        use_tp: bool = False,
        dtype: torch.dtype = torch.bfloat16,
        kv_cache_dtype: torch.dtype = None,
        device: Union[str, torch.device] = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        ),
        full_dialogue: bool = False,
    ):
        super().__init__(
            model_name, compile, quant, use_tp, dtype, kv_cache_dtype, device
        )
        assert "llava" in model_name.lower(), "This class is only for LLaVA models"
        self.full_dialogue = full_dialogue

    def apply_chat_template(
        self, conversation: DialogueForLLaVA, **kwargs
    ) -> Union[str, torch.Tensor]:
        raise NotImplementedError(
            "Chat template is applied in the generate method of the model."
        )

    @property
    def chat_template(self):
        raise NotImplementedError(
            "No Chat template for LLaVA models. Huggingface chat templage is used instead."
        )

    def dialogue_from_dialogue_vqa(self, dialogue: DialogueForVQA):
        """Version for VLM"""
        res = []
        if not self.full_dialogue:
            for message in dialogue:
                if message["role"] == "initial":
                    res.append(
                        {
                            "role": "system",
                            "content": [{"type": "text", "text": message["content"]}],
                        }
                    )
                elif message["role"] == "llm":
                    res.append(
                        {
                            "role": "user",
                            "content": [{"type": "text", "text": message["content"]}],
                        }
                    )
                elif message["role"] == "vlm":
                    res.append(
                        {
                            "role": "user",
                            "content": [{"type": "text", "text": message["content"]}],
                        }
                    )
            # Add the image in the first message:
            assert res[1]["role"] == "user"
            res[1]["content"].append({"type": "image"})
        else:
            res.append(
                {
                    "role": "system",
                    "content": [{"type": "text", "text": dialogue[0]["content"]}],
                }
            )
            res.append(
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": dialogue[-1]["content"]},
                        {"type": "image"},
                    ],
                }
            )
        return res

    def get_answer(
        self,
        dialogue: DialogueForVQA,
        image: Union[str, Image.Image, torch.Tensor],
        **sampling_kwargs
    ):
        prompt = self.dialogue_from_dialogue_vqa(dialogue)
        answer = self.generate(
            prompt=prompt,
            image=image,
            only_new_tokens=True,
            only_first_image_features=True,
            **sampling_kwargs
        )[0]
        dialogue.append({"role": "vlm", "content": answer})
        return dialogue


if __name__ == "__main__":
    # vqa_diag = [
    #     {
    #         "role": "initial",
    #         "content": "We are playing a game. I have selected an image. By asking me questions, you objective is to write the best captions possible to recover the original image from your caption. First, ask me one (and only one) direct questions about the image and wait for my answer. Ensure questions are closely tied to the image content. Emphasize questions that focus on intricate details, like recognizing objects, pinpointing positions, identifying colors, counting quantities, feeling moods, and more. Do NOT imagine, invent or infer anything. Ask me one question (and give only the question without other text). A short caption of the image is: A plate",
    #     },
    #     {
    #         "role": "llm",
    #         "content": "Is the tennis racket held by the man in his dominant hand?",
    #     },
    #     {
    #         "role": "vlm",
    #         "content": "Yes, the tennis racket is held by the man in his dominant hand. hopefully for a successful serve.",
    #     },
    # ]
    torch.manual_seed(42)
    vlm = LLaVAForVQA(compile=False, quant="int4")
    llm = Llama3ForVQA(compile=False, quant="int4")
    image = "/home/lucas/Documents/Datasets/ImageNet1K/ILSVRC/Data/CLS-LOC/val/ILSVRC2012_val_00018917.JPEG"

    initial_caption = vlm.generate(
        prompt="Give me a short sentence to describe this image",
        image=image,
        max_new_tokens=150,
        only_new_tokens=True,
    )[0]
    print("Initial caption: ", initial_caption)
    vqa_diag = [
        {
            "role": "initial",
            "content": "We are playing a game. I have selected an image. By asking me questions, you objective is to write the best captions possible to recover the original image from your caption. First, ask me one (and only one) direct questions about the image and wait for my answer. Ensure questions are closely tied to the image content. Emphasize questions that focus on intricate details, like recognizing objects, pinpointing positions, identifying colors, counting quantities, feeling moods, and more. Do NOT imagine, invent or infer anything. Ask me one question (and give only the question without other text). A short caption of the image is: "
            + initial_caption,
        }
    ]
    for i in range(2):
        vqa_diag = llm.get_question(vqa_diag, is_last=False, temperature=0.8, top_k=200)
        print(vqa_diag[-1])
        vqa_diag = vlm.get_answer(vqa_diag, image, temperature=0.8, top_k=200)
        print(vqa_diag[-1])
    vqa_diag[-1][
        "content"
    ] += ". Now, provide me a very detailed description of the image. It should be comprehensive, conversational, and use complete sentences. Provide context where necessary and maintain a certain tone. Incorporate information from answers into descriptive paragraphs. Do NOT imagine, invent or infer anything. Just output the description and do not say or write anything else. Be as detailed and descriptive as possible."
    vqa_diag = llm.get_question(vqa_diag, is_last=True, temperature=0.8, top_k=200)
    print(vqa_diag[-1])
