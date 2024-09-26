from typing import TypedDict, Sequence, Literal, List, Union, Optional, NotRequired
import os
import sys
import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from llm.tokenizer import TiktokenWrapper, get_tokenizer

## Base Dialogue class (based on Llama)
Role = Literal["system", "user", "assistant"]


class Message(TypedDict):
    role: Role
    content: str


Dialogue = Sequence[Message]


## Dialogue for VQA
RoleForVQA = Literal["initial", "llm", "vlm"]


class MessageForVQA(TypedDict):
    role: RoleForVQA
    content: str


DialogueForVQA = Sequence[MessageForVQA]


## Dialogue for LLaVA
TypeForLLaVA = Literal["text", "image"]


class ContentForLLaVA(TypedDict):
    type: TypeForLLaVA
    text: NotRequired[str]


class MessageForLLaVA(TypedDict):
    role: Role
    content: Sequence[ContentForLLaVA]


DialogueForLLaVA = Sequence[MessageForLLaVA]


### Adapted from https://github.com/meta-llama/llama3/blob/main/llama/tokenizer.py
class ChatFormat:
    """Format a dialogue into a chat prompt for a Tiktoken-based model."""

    def __init__(self, tokenizer: TiktokenWrapper):
        self.tokenizer = tokenizer

    def encode_header(self, message: Message) -> List[int]:
        tokens = []
        tokens.append(self.tokenizer.special_tokens["<|start_header_id|>"])
        tokens.extend(self.tokenizer.encode(message["role"], bos=False, eos=False))
        tokens.append(self.tokenizer.special_tokens["<|end_header_id|>"])
        tokens.extend(self.tokenizer.encode("\n\n", bos=False, eos=False))
        return tokens

    def encode_message(self, message: Message) -> List[int]:
        tokens = self.encode_header(message)
        if isinstance(message["content"], str):
            tokens.extend(
                self.tokenizer.encode(message["content"].strip(), bos=False, eos=False)
            )
        elif isinstance(message["content"], list):
            for item in message["content"]:
                if item["type"] == "text":
                    tokens.extend(
                        self.tokenizer.encode(
                            item["text"].strip(), bos=False, eos=False
                        )
                    )
                elif item["type"] == "image":
                    tokens.append(self.tokenizer.special_tokens["<image>"])
        else:
            raise ValueError("Invalid message content")
        tokens.append(self.tokenizer.special_tokens["<|eot_id|>"])
        return tokens

    def encode_dialogue_prompt(
        self, dialogue: Dialogue, return_tensor: bool = True
    ) -> Union[List[int], torch.Tensor]:
        tokens = []
        tokens.append(self.tokenizer.special_tokens["<|begin_of_text|>"])
        for message in dialogue:
            tokens.extend(self.encode_message(message))
        tokens.extend(self.encode_header({"role": "assistant", "content": ""}))
        if return_tensor:
            return torch.tensor(tokens, dtype=torch.int32, device="cuda")
        return tokens


if __name__ == "__main__":
    from pathlib import Path

    checkpoint_path = Path(
        # "/home/lucas/Documents/checkpoints/meta-llama/Meta-Llama-3-8B"
        "/home/lucas/Documents/checkpoints/llava-hf/llama3-llava-next-8b-hf/"
    )
    model_name = "Meta-Llama-3-8B"
    tokenizer = get_tokenizer(model_name, checkpoint_path)
    chat = ChatFormat(tokenizer)
    conv = [
        {"role": "user", "content": "Hello!"},
        {"role": "assistant", "content": "Hi! How can I help you today?"},
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {
                    "type": "text",
                    "text": "Give me a very descriptive caption of this image.\n",
                },
            ],
        },
    ]
    encoded = chat.encode_dialogue_prompt(conv, return_tensor=False)
    print(encoded)
    decoded = tokenizer.decode(encoded, skip_special_tokens=False)
    print(decoded)
