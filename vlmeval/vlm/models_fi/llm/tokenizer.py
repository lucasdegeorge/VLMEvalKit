import os
import sentencepiece as spm
import tiktoken
import torch
import numpy as np
from tiktoken.load import load_tiktoken_bpe
from transformers import AutoTokenizer
from pathlib import Path
from typing import Dict, List, Union


class TokenizerInterface:
    def __init__(self, tokenizer_path):
        self.tokenizer_path = tokenizer_path

    def encode(self, text):
        raise NotImplementedError("This method should be overridden by subclasses.")

    def decode(self, tokens):
        raise NotImplementedError("This method should be overridden by subclasses.")

    def bos_id(self):
        raise NotImplementedError("This method should be overridden by subclasses.")

    def eos_id(self):
        raise NotImplementedError("This method should be overridden by subclasses.")


class SentencePieceWrapper(TokenizerInterface):
    def __init__(self, tokenizer_path):
        super().__init__(tokenizer_path)
        self.processor = spm.SentencePieceProcessor(str(tokenizer_path))
        self.stop_tokens = [self.processor.eos_id()]

    def encode(self, text: str, bos: bool = None, eos: bool = None) -> List[int]:
        assert isinstance(text, str), "text must be a string. Got: {}".format(text)
        tokens = self.processor.EncodeAsIds(text)
        if bos:
            tokens = [self.bos_id()] + tokens
        if eos:
            tokens = tokens + [self.eos_id()]
        return tokens

    def decode(self, tokens: List[int]) -> str:
        assert isinstance(tokens, list) and all(
            isinstance(t, int) for t in tokens
        ), "tokens must be a list of integers. Got: {}".format(type(tokens))
        return self.processor.DecodeIds(tokens)

    def batch_encode(
        self,
        texts: List[str],
        bos: List[bool] = None,
        eos: List[bool] = None,
        padding_side: str = "left",
        device: Union[str, torch.device] = "cuda",
    ) -> List[List[int]]:
        assert isinstance(texts, list) and all(
            isinstance(t, str) for t in texts
        ), "texts must be a list of strings. Got: {}".format(type(texts))
        if bos is not None:
            assert len(texts) == len(bos)
        if eos is not None:
            assert len(texts) == len(eos)

        batched_tokens = self.processor.EncodeAsIds(texts)

        for i in range(len(texts)):
            if bos and bos[i]:
                batched_tokens[i] = [self.bos_id()] + batched_tokens[i]
            if eos and eos[i]:
                batched_tokens[i] = batched_tokens[i] + [self.eos_id()]

        max_len = max(len(tokens) for tokens in batched_tokens)

        if padding_side == "left":
            left_pad_mask_pos = torch.tensor(
                [max_len - len(tokens) for tokens in batched_tokens],
                dtype=torch.int,
                device=device,
            )
            batched_tokens = [
                [self.pad_id()] * (max_len - len(tokens)) + tokens
                for tokens in batched_tokens
            ]
        elif padding_side == "right":
            batched_tokens = [
                tokens + [self.pad_id()] * (max_len - len(tokens))
                for tokens in batched_tokens
            ]
            left_pad_mask_pos = None
        else:
            raise ValueError("padding_side must be either 'left' or 'right'.")

        return (
            torch.tensor(batched_tokens, dtype=torch.int, device=device),
            left_pad_mask_pos,
        )

    def batch_decode(self, tokens: Union[List[List[int]], torch.Tensor]) -> List[str]:
        if isinstance(tokens, torch.Tensor):
            tokens = tokens.tolist()
        assert isinstance(tokens, list) and all(
            isinstance(t, list) for t in tokens
        ), "tokens must be a list of list of integers. Got: {}".format(type(tokens))
        return [self.decode(token) for token in tokens]

    def bos_id(self):
        return self.processor.bos_id()

    def eos_id(self):
        return self.processor.eos_id()

    def pad_id(self):
        return self.processor.pad_id()


class TiktokenWrapper(TokenizerInterface):
    """
    Tokenizing and encoding/decoding text using the Tiktoken tokenizer.
    """

    special_tokens: Dict[str, int]

    num_reserved_special_tokens = 256

    pat_str = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"  # noqa: E501

    def __init__(self, tokenizer_path):
        super().__init__(tokenizer_path)
        assert os.path.isfile(tokenizer_path), str(tokenizer_path)
        mergeable_ranks = load_tiktoken_bpe(str(tokenizer_path))
        num_base_tokens = len(mergeable_ranks)
        special_tokens = [
            "<|begin_of_text|>",
            "<|end_of_text|>",
            "<|reserved_special_token_0|>",
            "<|reserved_special_token_1|>",
            "<|reserved_special_token_2|>",
            "<|reserved_special_token_3|>",
            "<|start_header_id|>",
            "<|end_header_id|>",
            "<|reserved_special_token_4|>",
            "<|eot_id|>",  # end of turn
        ] + [
            f"<|reserved_special_token_{i}|>"
            for i in range(5, self.num_reserved_special_tokens - 5)
        ]
        self.special_tokens = {
            token: num_base_tokens + i for i, token in enumerate(special_tokens)
        }
        self.stop_tokens = [
            self.special_tokens["<|end_of_text|>"],
            self.special_tokens["<|eot_id|>"],
        ]
        self.model = tiktoken.Encoding(
            name=Path(tokenizer_path).name,
            pat_str=self.pat_str,
            mergeable_ranks=mergeable_ranks,
            special_tokens=self.special_tokens,
        )
        # BOS / EOS token IDs
        self._bos_id: int = self.special_tokens["<|begin_of_text|>"]
        self._eos_id: int = self.special_tokens["<|end_of_text|>"]

    def encode(self, text: str, bos: bool = None, eos: bool = None) -> List[int]:
        assert isinstance(text, str), "text must be a string. Got: {}".format(text)
        tokens = self.model.encode(text)
        if bos:
            tokens = [self._bos_id] + tokens
        if eos:
            tokens.append(self._eos_id)
        return tokens

    def decode(self, tokens: List[int]) -> str:
        assert isinstance(tokens, list) and all(
            isinstance(t, int) for t in tokens
        ), "tokens must be a list of integers. Got: {}".format(type(tokens))
        return self.model.decode(tokens)

    def batch_encode(
        self,
        texts: List[str],
        bos: List[bool] = None,
        eos: List[bool] = None,
        padding_side: str = "left",
        device: Union[str, torch.device] = "cuda",
    ) -> List[List[int]]:
        assert isinstance(texts, list) and all(
            isinstance(t, str) for t in texts
        ), "texts must be a list of strings. Got: {}".format(type(texts))

        batched_tokens = self.model.encode_batch(texts)

        for i in range(len(texts)):
            if bos and bos[i]:
                batched_tokens[i] = [self._bos_id] + batched_tokens[i]
            if eos and eos[i]:
                batched_tokens[i] = batched_tokens[i] + [self._eos_id]

        max_len = max(len(tokens) for tokens in batched_tokens)

        if padding_side == "left":
            left_pad_mask_pos = torch.tensor(
                [max_len - len(tokens) for tokens in batched_tokens],
                dtype=torch.int,
                device=device,
            )
            batched_tokens = [
                [self.pad_id()] * (max_len - len(tokens)) + tokens
                for tokens in batched_tokens
            ]
        elif padding_side == "right":
            batched_tokens = [
                tokens + [self.pad_id()] * (max_len - len(tokens))
                for tokens in batched_tokens
            ]
            left_pad_mask_pos = None
        else:
            raise ValueError("padding_side must be either 'left' or 'right'.")

        return (
            torch.tensor(batched_tokens, dtype=torch.int, device=device),
            left_pad_mask_pos,
        )

    def batch_decode(self, tokens: Union[List[List[int]], torch.Tensor]) -> List[str]:
        if isinstance(tokens, torch.Tensor):
            tokens = tokens.tolist()
        assert isinstance(tokens, list) and all(
            isinstance(t, list) for t in tokens
        ), "tokens must be a list of list of integers. Got: {}".format(type(tokens))
        return self.model.decode_batch(tokens)

    def bos_id(self):
        return self._bos_id

    def eos_id(self):
        return self._eos_id

    def pad_id(self):
        return self._eos_id


class HuggingFaceWrapper(TokenizerInterface):
    def __init__(self, tokenizer_path: Union[str, Path] = None, hf_folder: str = None):
        assert (
            tokenizer_path is not None or hf_folder is not None
        ), "Either tokenizer_path or hf_folder must be provided."
        super().__init__(tokenizer_path)
        self.tokenizer = None

        if tokenizer_path is not None:
            if str(tokenizer_path).endswith(".json"):
                assert os.path.isfile(tokenizer_path), str(tokenizer_path)
                self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path.parent)
                print("Tokenizer loaded from json file with Huggingface.")
            elif str(tokenizer_path).endswith(".model"):
                raise ValueError(
                    "HFWrapper does not support '.model' SentencePiece tokenizers."
                )

        if hf_folder is not None and self.tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(hf_folder)
            print("Tokenizer loaded from pretrained Huggingface Tokenizer online.")

        if self.tokenizer is None:
            raise ValueError("Unable to load tokenizer from the provided paths.")

        num_base_tokens = 128000
        num_reserved_special_tokens = 256
        special_tokens = (
            [
                "<|begin_of_text|>",
                "<|end_of_text|>",
                "<|reserved_special_token_0|>",
                "<|reserved_special_token_1|>",
                "<|reserved_special_token_2|>",
                "<|reserved_special_token_3|>",
                "<|start_header_id|>",
                "<|end_header_id|>",
                "<|reserved_special_token_4|>",
                "<|eot_id|>",  # end of turn
            ]
            + [
                f"<|reserved_special_token_{i}|>"
                for i in range(5, num_reserved_special_tokens - 5)
            ]
            + [
                "<image>",
                "<pad>",
            ]
        )
        self.special_tokens = {
            token: num_base_tokens + i for i, token in enumerate(special_tokens)
        }

        self.stop_tokens = [
            self.tokenizer.eos_token_id,
            # tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]

    def bos_id(self):
        return self.tokenizer.bos_token_id

    def eos_id(self):
        return self.tokenizer.eos_token_id

    def pad_id(self):
        return (
            self.tokenizer.pad_token_id
            if self.tokenizer.pad_token_id is not None
            else self.tokenizer.eos_token_id
        )

    def encode(self, text: str, bos: bool = None, eos: bool = None) -> List[int]:
        assert isinstance(text, str), "text must be a string. Got: {}".format(text)
        tokens = self.tokenizer.encode(text, add_special_tokens=False)
        if bos:
            tokens = [self.bos_id()] + tokens
        if eos:
            tokens.append(self.eos_id())
        return tokens

    def decode(self, tokens: List[int], skip_special_tokens: bool = True) -> str:
        assert isinstance(tokens, list) and all(
            isinstance(t, int) for t in tokens
        ), "tokens must be a list of integers. Got: {}".format(type(tokens))
        return self.tokenizer.decode(tokens, skip_special_tokens=skip_special_tokens)

    def batch_encode(
        self,
        texts: List[str],
        bos: List[bool] = None,
        eos: List[bool] = None,
        padding_side: str = "left",
        device: Union[str, torch.device] = "cuda",
    ):
        assert isinstance(texts, list) and all(
            isinstance(t, str) for t in texts
        ), "texts must be a list of strings. Got: {}".format(type(texts))
        if bos is not None:
            assert len(texts) == len(bos)
        if eos is not None:
            assert len(texts) == len(eos)

        batched_tokens = self.tokenizer(texts, add_special_tokens=False)["input_ids"]

        for i in range(len(texts)):
            if bos and bos[i]:
                batched_tokens[i] = [self.bos_id()] + batched_tokens[i]
            if eos and eos[i]:
                batched_tokens[i] = batched_tokens[i] + [self.eos_id()]

        max_len = max(len(tokens) for tokens in batched_tokens)

        if padding_side == "left":
            left_pad_mask_pos = torch.tensor(
                [max_len - len(tokens) for tokens in batched_tokens],
                dtype=torch.int,
                device=device,
            )
            batched_tokens = [
                [self.pad_id()] * (max_len - len(tokens)) + tokens
                for tokens in batched_tokens
            ]
        elif padding_side == "right":
            batched_tokens = [
                tokens + [self.pad_id()] * (max_len - len(tokens))
                for tokens in batched_tokens
            ]
            left_pad_mask_pos = None
        else:
            raise ValueError("padding_side must be either 'left' or 'right'.")

        return (
            torch.tensor(batched_tokens, dtype=torch.int, device=device),
            left_pad_mask_pos,
        )

    def batch_decode(
        self,
        tokens: Union[List[List[int]], torch.Tensor],
        skip_special_tokens: bool = True,
    ) -> List[str]:
        if isinstance(tokens, torch.Tensor):
            tokens = tokens.tolist()
        assert isinstance(tokens, list) and all(
            isinstance(t, list) for t in tokens
        ), "tokens must be a list of list of integers. Got: {}".format(type(tokens))
        return self.tokenizer.batch_decode(
            tokens, skip_special_tokens=skip_special_tokens
        )


def get_tokenizer(
    model_name: str, checkpoint_folder: Union[str, Path] = None, hf_folder: str = None
) -> TokenizerInterface:
    """
    Function to get the appropriate tokenizer based on the model name.
    Args:
    - model_name (str): The name of the model, used to determine the tokenizer type.
    - checkpoint_folder (str): The path to the folder containing the model checkpoint.
    - hf_folder (str, Optional): The name of the HF model directory.
    Returns:
    - TokenizerInterface: An instance of a tokenizer.
    """
    try:
        tokenizer_path = Path(checkpoint_folder) / "tokenizer.model"
        assert tokenizer_path.exists()
        if "Llama-3" in str(model_name):
            return TiktokenWrapper(tokenizer_path)
        else:
            return SentencePieceWrapper(tokenizer_path)
    except AssertionError:
        try:
            tokenizer_path = Path(checkpoint_folder) / "tokenizer.json"
            return HuggingFaceWrapper(tokenizer_path, hf_folder)
        except ValueError or AssertionError:
            raise ValueError("Invalid tokenizer path or model name.")


if __name__ == "__main__":
    tokenizer_path = "/home/lucas/Documents/checkpoints/llava-hf/llama3-llava-next-8b-hf"  # "/home/lucas/Documents/checkpoints/meta-llama/Meta-Llama-3.1-8B-Instruct"
    model_name = "Llava"
    texts = ["Hello, my name is Lucas", "Hello World"]

    tokenizer = get_tokenizer(
        model_name, tokenizer_path, "llava-hf/llama3-llava-next-8b-hf"
    )
    print("tokenizer", tokenizer)
    encoded_tokens, left_pad_mask_pos = tokenizer.batch_encode(texts)

    print("Encoded tokens:", encoded_tokens)
    print("left_pad_mask_pos:", left_pad_mask_pos)
