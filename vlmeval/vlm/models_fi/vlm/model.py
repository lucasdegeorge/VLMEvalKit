import torch
from pathlib import Path
import time
from typing import Tuple, Union, List
import contextlib
from PIL import Image
import numpy as np
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from vlm.image_encoder.image_encoder_hf import (
    HFImageEncoder,
    HFLlavaNextImageEncoder,
    HFPaliGemmaImageEncoder,
)
from vlm.utils import generate_batch_of_vlm_prompt
from llm.utils import sample, remove_all_backward_hooks
from llm.tokenizer import get_tokenizer
from llm.architecture.base import Transformer
from llm.architecture.gemma import TransformerForGemma

from llm.tensor_parallel import (
    maybe_init_dist,
    initialize_model_parallel,
    apply_tp,
    get_model_parallel_rank,
    get_data_parallel_rank,
    get_data_parallel_world_size,
)

# import torch._dynamo.config
# torch._dynamo.config.capture_scalar_outputs = True


class VLMBase:
    def __init__(
        self,
        model_name: str,
        compile: bool = True,
        quant: str = "none",
        use_tp: bool = False,
        dtype: torch.dtype = torch.bfloat16,
        kv_cache_dtype: torch.dtype = None,
        device: Union[str, torch.device] = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        ),
    ) -> None:
        self.model_name = model_name
        self.compile = compile
        self.device = (
            device
            if device is not None
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.dtype = dtype if dtype is not None else torch.bfloat16
        torch.set_default_device(self.device)
        torch.set_default_dtype(self.dtype)
        self.kv_cache_dtype = (
            kv_cache_dtype if kv_cache_dtype is not None else self.dtype
        )

        rank = maybe_init_dist()
        self.use_tp = use_tp and rank is not None
        tp_size = 1
        if self.use_tp:
            tp_size = torch.distributed.get_world_size()
            initialize_model_parallel(tp_size)

        ## Image encoder
        self.image_encoder = self.load_image_model()

        ## Text encoder
        self.text_model = self.load_text_model(quant, use_tp=use_tp)
        self.tokenizer = get_tokenizer(model_name, self.text_checkpoint_path, None)
        self.eos_id = self.tokenizer.eos_id()
        self.pad_id = self.tokenizer.pad_id()

    def load_image_model(self):
        if "hf" in self.model_name:
            if "llava" in self.model_name:
                image_encoder = HFLlavaNextImageEncoder.from_name(
                    self.model_name,
                    do_delete_model=True,
                    device=self.device,
                    dtype=self.dtype,
                )
            elif "paligemma" in self.model_name:
                image_encoder = HFPaliGemmaImageEncoder.from_name(
                    self.model_name,
                    do_delete_model=True,
                    device=self.device,
                    dtype=self.dtype,
                )
            else:
                print(
                    f'Image encoder not found for model "{self.model_name}". Using default. Strange comportment may occur.'
                )
                image_encoder = HFImageEncoder.from_name(self.model_name)
            return image_encoder
        else:
            raise NotImplementedError(
                "Image encoder without HF support not implemented yet"
            )

    def load_text_model(self, quant: str = "none", use_tp: bool = False):
        """Similar to 'load_model' from LLMBase."""
        tick = time.time()
        with torch.device("meta"):
            model = Transformer.from_name(self.model_name)
            model.vocab_parallel = False  ## TODO: check if this is needed
            self.text_checkpoint_path = model.config.checkpoint_path
            if isinstance(self.text_checkpoint_path, str):
                self.text_checkpoint_path = Path(self.text_checkpoint_path)

        if quant == "none":
            self.model_path = self.text_checkpoint_path / "model.pth"
            assert self.model_path.is_file(), str(self.model_path)

        elif quant == "int8":
            self.model_path = self.text_checkpoint_path / "model_int8.pth"
            assert self.model_path.is_file(), str(self.model_path)
            from llm.quantization import WeightOnlyInt8QuantHandler

            simple_quantizer = WeightOnlyInt8QuantHandler(model)
            model = simple_quantizer.convert_for_runtime()

        elif quant == "int4":
            self.model_path = self.text_checkpoint_path / "model_int4.g32.pth"
            assert self.model_path.is_file(), str(self.model_path)
            path_comps = self.model_path.name.split(".")
            groupsize = int(path_comps[-2][1:])
            from llm.quantization import WeightOnlyInt4QuantHandler

            simple_quantizer = WeightOnlyInt4QuantHandler(model, groupsize)
            model = simple_quantizer.convert_for_runtime()

        else:
            raise ValueError(f"Invalid quantization type: {quant}")

        checkpoint = torch.load(str(self.model_path), mmap=True, weights_only=True)
        model.load_state_dict(checkpoint, assign=True)

        if use_tp:
            apply_tp(model)

        model = model.to(device=self.device, dtype=self.dtype)
        print(f"Model loaded in {time.time() - tick:.02f} seconds")
        return model.eval()

    def tokenize(self, string: str, bos: bool = True) -> torch.Tensor:
        """Useless right know with HF image encoder"""
        raise NotImplementedError("Useless right know with HF image encoder")
        tokens = self.tokenizer.encode(string)
        if bos:
            tokens = [self.tokenizer.bos_id()] + tokens
        return torch.tensor(tokens, dtype=torch.int, device=self.device)

    def batch_tokenize(
        self, strings: list, bos: List[bool] = None, eos: List[bool] = None
    ) -> torch.Tensor:
        """Useless right know with HF image encoder"""
        raise NotImplementedError("Useless right know with HF image encoder")
        assert isinstance(strings, list), "Input must be a list of strings."
        assert all(
            isinstance(s, str) for s in strings
        ), "Input must be a list of strings."
        return self.tokenizer.batch_encode(strings, bos=bos, eos=eos)

    def decode_one_token(
        self,
        x: torch.Tensor,
        input_pos: torch.Tensor,
        left_pad_mask_pos: torch.Tensor,
        **sampling_kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # input_pos: [B, 1]
        assert input_pos.shape[-1] == 1
        logits = self.text_model(
            idx=x, input_pos=input_pos, embeds=None, left_pad_mask_pos=left_pad_mask_pos
        )
        return sample(logits, self.text_model.vocab_parallel, **sampling_kwargs)

    def decode_n_tokens(  ## TODO: check for call-back !
        self,
        cur_token: torch.Tensor,
        input_pos: torch.Tensor,
        nb_new_tokens: int,
        stop_first_eos: bool = True,
        **sampling_kwargs,
    ):
        new_tokens, new_probs = [], []
        for i in range(nb_new_tokens):
            # with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH):
            with torch.backends.cuda.sdp_kernel(
                enable_flash=False, enable_mem_efficient=False, enable_math=True
            ):
                next_token, next_prob = self.decode_one_token(
                    cur_token, input_pos, left_pad_mask_pos=None, **sampling_kwargs
                )
                input_pos += 1
                new_tokens.append(next_token.clone())
                new_probs.append(next_prob.clone())
                cur_token = next_token.view(1, -1)
                if stop_first_eos and next_token in self.tokenizer.stop_tokens:
                    # print(f"Found EOS token at position {i}")
                    break
        return new_tokens, new_probs, i

    def batch_decode_n_tokens(
        self,
        cur_token: torch.Tensor,
        input_pos: torch.Tensor,
        left_pad_mask_pos: torch.Tensor,
        nb_new_tokens: int,
        stop_first_eos: bool = True,
        **sampling_kwargs,
    ):
        new_tokens, new_probs = [], []
        eos_flag = None
        eos_positions = torch.zeros(
            cur_token.size(0), dtype=torch.long, device=cur_token.device
        )

        if stop_first_eos:
            eos_flag = torch.zeros_like(
                cur_token, dtype=torch.bool, device=cur_token.device
            )

        for i in range(nb_new_tokens):
            # with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH):
            with torch.backends.cuda.sdp_kernel(
                enable_flash=False, enable_mem_efficient=False, enable_math=True
            ):
                next_token, next_prob = self.decode_one_token(
                    cur_token, input_pos, left_pad_mask_pos, **sampling_kwargs
                )
            input_pos += 1
            new_tokens.append(next_token.clone().view(-1, 1))
            new_probs.append(next_prob.clone().view(-1, 1))
            cur_token = next_token.view(-1, 1)

            if eos_flag is not None:
                eos_flag = eos_flag | (next_token == self.tokenizer.eos_id())
                eos_positions[
                    (eos_flag & (eos_positions == 0)).nonzero(as_tuple=False)
                ] = i

            if stop_first_eos and eos_flag is not None and eos_flag.all():
                # print(f"Found EOS token at position {i}")
                break

        return new_tokens, new_probs, i

    def prefill(
        self,
        x: torch.Tensor,
        input_pos: torch.Tensor,
        embeds: torch.Tensor,
        left_pad_mask_pos: torch.Tensor,
        **sampling_kwargs,
    ) -> torch.Tensor:
        # input_pos: [B, S]
        logits = self.text_model(x, input_pos, embeds, left_pad_mask_pos)
        return sample(logits, self.text_model.vocab_parallel, **sampling_kwargs)

    @torch.no_grad()
    def generate(
        self,
        prompt: Union[torch.Tensor, str],
        image: Union[str, Image.Image, torch.Tensor],
        max_new_tokens: int = 200,
        do_decode: bool = True,
        stop_first_eos: bool = True,
        only_new_tokens: bool = False,
        remove_image_tokens: bool = True,
        only_first_image_features: bool = False,
        **sampling_kwargs,
    ) -> torch.Tensor:

        if self.compile:  # TO DO: check and if add compile_prefill
            self.decode_one_token = torch.compile(
                self.decode_one_token, mode="reduce-overhead", fullgraph=True
            )

        # if do_decode and not remove_image_tokens:
        #     raise ValueError("Cannot return decoded sequence with image tokens")

        # Prompts and Embeddings

        ## Without HF image encoder
        # if isinstance(prompt, str):
        #     oprompt = self.tokenize(prompt, bos=True)
        # text_embeddings = self.text_model.tok_embeddings(oprompt)
        # print("initial prompt", oprompt.shape, oprompt.dtype)

        # With HF image encoder
        inputs = self.image_encoder.preprocess_prompt_image(prompt, image)
        prompt = inputs["input_ids"]
        text_embeddings = self.text_model.tok_embeddings(prompt)
        prompt = prompt.squeeze(0)  # Remove batch dimension in generate

        ti_embeddings, prompt, image_embeddings = (
            self.image_encoder.merge_text_image_embeddings(
                inputs, text_embeddings.to(self.dtype), only_first_image_features
            )
        )
        ti_embeddings = ti_embeddings.to(self.dtype)
        prompt = prompt.squeeze(0)  # Remove batch dimension in generate

        # Generation
        T = prompt.size(0)
        T_max = T + max_new_tokens
        max_seq_length = min(T_max, self.text_model.config.block_size)

        self.text_model.setup_caches(
            max_batch_size=1,
            max_seq_length=max_seq_length,
            prompt_size=T,
            device=prompt.device,
            kv_dtype=self.kv_cache_dtype,
        )
        input_pos = torch.arange(0, T, device=prompt.device)

        # with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH):
        with torch.backends.cuda.sdp_kernel(
            enable_flash=False, enable_mem_efficient=False, enable_math=True
        ):
            next_token, _ = self.prefill(
                prompt.view(1, -1),
                input_pos=input_pos,
                embeds=ti_embeddings,
                left_pad_mask_pos=None,
                **sampling_kwargs,
            )

        # Generate
        input_pos = torch.tensor([T], device=prompt.device, dtype=torch.int)
        generated_tokens, _, nb_generated_tokens = self.decode_n_tokens(
            next_token.view(1, -1),
            input_pos,
            nb_new_tokens=max_new_tokens - 1,
            stop_first_eos=stop_first_eos,
            **sampling_kwargs,
        )

        # Fill an empty tensor with the generated tokens
        T_new = T + len(generated_tokens) + 1
        empty = torch.empty(T_new, dtype=prompt.dtype, device=prompt.device)
        empty[:T] = prompt
        seq = empty
        seq[T] = next_token
        seq[T + 1 :] = torch.cat(generated_tokens)

        if only_new_tokens:
            seq = seq[T:]
        # elif remove_image_tokens:
        #     seq = seq[self.text_model.config.n_image_tokens :]

        if do_decode:
            seq = self.tokenizer.decode(seq.tolist(), skip_special_tokens=True)

        return seq, nb_generated_tokens

    @torch.no_grad()
    def batch_generate(
        self,
        prompts: Union[torch.Tensor, List[str]],
        images: Union[str, Image.Image, torch.Tensor],
        left_pad_mask_pos: torch.Tensor,
        max_new_tokens: int = 200,
        do_decode: bool = True,
        stop_first_eos: bool = True,
        only_new_tokens: bool = False,
        remove_image_tokens: bool = True,
        only_first_image_features: bool = False,
        **sampling_kwargs,
    ) -> Union[torch.Tensor, str]:

        if self.compile:
            self.decode_one_token = torch.compile(
                self.decode_one_token, mode="reduce-overhead", fullgraph=True
            )
            # self.prefill = torch.compile(
            #     self.prefill, mode="reduce-overhead", fullgraph=True
            # )

        if self.use_tp:
            torch.distributed.barrier()
            dp_rank = get_data_parallel_rank()
            tp_rank = get_model_parallel_rank()
            dp_size = get_data_parallel_world_size()

        if self.compile:
            remove_all_backward_hooks(self.text_model)

        # Prompts and Embeddings
        inputs = self.image_encoder.batch_preprocess_prompt_image(prompts, images)
        prompt = inputs["input_ids"]
        text_embeddings = self.text_model.tok_embeddings(prompt)

        ti_embeddings, prompt, image_embeddings = (
            self.image_encoder.merge_text_image_embeddings(
                inputs, text_embeddings.to(self.dtype), only_first_image_features
            )
        )
        ti_embeddings = ti_embeddings.to(self.dtype)

        # Generation
        B = prompt.size(0)
        T = prompt.size(1)
        T_max = T + max_new_tokens
        max_seq_length = min(T_max, self.text_model.config.block_size)

        self.text_model.setup_caches(
            max_batch_size=B,
            max_seq_length=max_seq_length,
            prompt_size=T,
            device=prompt.device,
            kv_dtype=self.kv_cache_dtype,
        )
        input_pos = torch.arange(0, T, device=prompt.device)

        # with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH):
        with torch.backends.cuda.sdp_kernel(
            enable_flash=False, enable_mem_efficient=False, enable_math=True
        ):
            next_token, _ = self.prefill(
                prompt.view(B, -1),
                input_pos=input_pos,
                embeds=ti_embeddings,
                left_pad_mask_pos=left_pad_mask_pos,
                **sampling_kwargs,
            )

        # Generate
        input_pos = torch.tensor([T], device=prompt.device, dtype=torch.int)
        generated_tokens, _, nb_generated_tokens = self.batch_decode_n_tokens(
            cur_token=next_token.view(B, -1),
            input_pos=input_pos,
            left_pad_mask_pos=left_pad_mask_pos,
            nb_new_tokens=max_new_tokens - 1,
            stop_first_eos=stop_first_eos,
            **sampling_kwargs,
        )
        generated_tokens = torch.cat(generated_tokens, dim=-1).view(B, -1)

        # Fill an empty tensor with the generated tokens
        T_new = T + max_new_tokens
        empty = torch.empty((B, T_new), dtype=prompt.dtype, device=prompt.device)
        empty[:, :T] = prompt
        empty[:, T:] = self.pad_id
        seq = empty
        seq[:, T] = next_token.view(B)
        seq[:, T + 1 : T + 1 + generated_tokens.size(1)] = generated_tokens

        if only_new_tokens:
            seq = seq[:, T:]

        if do_decode:
            # print(seq.tolist())
            seq[seq == -1] = 1  # convert improper tokens to ''
            seq[seq == 198] = 1  # convert improper tokens to ''
            seq = self.tokenizer.batch_decode(seq.tolist())

        return seq, nb_generated_tokens

    def benchmark_tok_per_s(
        self,
        prompt: str,
        image: Union[str, Image.Image],
        max_new_tokens: int,
        nb_samples: int = 100,
        **sampling_kwargs,
    ) -> float:
        tokens_per_s = list()
        for i in range(-1, nb_samples):
            t0 = time.perf_counter()
            output, nb_generated_tokens = self.generate(
                prompt,
                image,
                max_new_tokens,
                do_decode=False,
                stop_first_eos=False,
                remove_image_tokens=True,
                **sampling_kwargs,
            )
            if i == -1:
                print(f"Compilation time: {time.perf_counter() - t0:.2f} seconds")
                continue
            else:
                tokens_generated = nb_generated_tokens
                tokens_per_s.append(tokens_generated / (time.perf_counter() - t0))
        print(f"Average tokens per second: {np.mean(tokens_per_s):.2f}")
        return np.mean(tokens_per_s)

    def benchmark_batch_tok_per_sec(
        self,
        batch_size: int,
        max_new_tokens: int,
        nb_samples: int = 5,
        **sampling_kwargs,
    ) -> float:
        tokens_per_s = list()
        for i in range(-1, nb_samples):
            images, prompts = generate_batch_of_vlm_prompt(batch_size)
            t0 = time.perf_counter()
            output, nb_generated_tokens = self.batch_generate(
                prompts=prompts,
                images=images,
                left_pad_mask_pos=None,
                max_new_tokens=max_new_tokens,
                do_decode=False,
                stop_first_eos=False,
                only_first_image_features=True**sampling_kwargs,
            )
            if i == -1:
                print(f"Compilation time: {time.perf_counter() - t0:.2f} seconds")
                continue
            else:
                # print(self.tokenizer.decode(output.tolist()))
                # print("-----------------------")
                print(f"Generation time: {(time.perf_counter() - t0):.2f} seconds")
                tokens_generated_sec = nb_generated_tokens * output.size(0)
                tokens_per_s.append(tokens_generated_sec / (time.perf_counter() - t0))
        print(f"Average tokens per second: {np.mean(tokens_per_s):.2f}")
        return np.mean(tokens_per_s)


class GemmaVLMBase(VLMBase):
    def __init__(
        self,
        model_name: str,
        compile: bool = True,
        quant: str = "none",
        dtype: torch.dtype = torch.bfloat16,
        device: Union[str, torch.device] = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        ),
    ) -> None:
        super().__init__(model_name, compile, quant, dtype, device)

    def load_text_model(self, quant: str = "none"):
        """Similar to 'load_model' from GemmaBase."""
        tick = time.time()
        with torch.device("meta"):
            model = TransformerForGemma.from_name(self.model_name)
            self.text_checkpoint_path = model.config.checkpoint_path
            if isinstance(self.text_checkpoint_path, str):
                self.text_checkpoint_path = Path(self.text_checkpoint_path)

        if quant == "none":
            self.model_path = self.text_checkpoint_path / "model.pth"
            assert self.model_path.is_file(), str(self.model_path)

        elif quant == "int8":
            self.model_path = self.text_checkpoint_path / "model_int8.pth"
            assert self.model_path.is_file(), str(self.model_path)
            from llm.quantization import WeightOnlyInt8QuantHandler

            simple_quantizer = WeightOnlyInt8QuantHandler(model)
            model = simple_quantizer.convert_for_runtime()

        elif quant == "int4":
            self.model_path = self.text_checkpoint_path / "model_int4.g32.pth"
            assert self.model_path.is_file(), str(self.model_path)
            path_comps = self.model_path.name.split(".")
            groupsize = int(path_comps[-2][1:])
            from llm.quantization import WeightOnlyInt4QuantHandler

            simple_quantizer = WeightOnlyInt4QuantHandler(model, groupsize)
            model = simple_quantizer.convert_for_runtime()

        else:
            raise ValueError(f"Invalid quantization type: {quant}")

        checkpoint = torch.load(str(self.model_path), mmap=True, weights_only=True)
        model.load_state_dict(checkpoint, assign=True)

        model = model.to(device=self.device, dtype=self.dtype)
        print(f"Model loaded in {time.time() - tick:.02f} seconds")
        return model.eval()

    @torch.no_grad()
    def generate(
        self,
        prompt: Union[torch.Tensor, str],
        image: Union[str, Image.Image, torch.Tensor],
        max_new_tokens: int,
        do_decode: bool = True,
        stop_first_eos: bool = True,
        only_new_tokens: bool = False,
        remove_image_tokens: bool = True,
        **sampling_kwargs,
    ) -> torch.Tensor:

        # Preprocessing
        prof = contextlib.nullcontext()

        if self.compile:  # TO DO: check and if add compile_prefill
            self.decode_one_token = torch.compile(
                self.decode_one_token, mode="reduce-overhead", fullgraph=True
            )

        # if do_decode and not remove_image_tokens:
        #     raise ValueError("Cannot return decoded sequence with image tokens")

        # Prompts and Embeddings
        if isinstance(prompt, str):
            prompt = self.tokenize(prompt, bos=True)

        text_embeddings = self.text_model.tok_embeddings(prompt)
        ti_embeddings, image_embeddings = (
            self.image_encoder.merge_text_image_embeddings(
                image, text_embeddings=text_embeddings
            )
        )

        image_prompt = torch.full(
            (self.text_model.config.vision_args.n_image_tokens,),
            self.text_model.config.vision_args.image_token_index,
            device=prompt.device,
            dtype=prompt.dtype,
        )
        prompt = torch.cat([image_prompt, prompt], dim=0)

        # Generation
        with prof:
            T = prompt.size(0)
            T_max = T + max_new_tokens
            max_seq_length = min(T_max, self.text_model.config.block_size)

            ## New line for PaliGemma models: prompt_size for prefixLM mask
            self.text_model.setup_caches(
                max_batch_size=1,
                max_seq_length=max_seq_length,
                prompt_size=T,
                device=prompt.device,
            )
            input_pos = torch.arange(0, T, device=prompt.device)
            next_token = self.prefill(
                prompt.view(1, -1),
                input_pos=input_pos,
                embeds=ti_embeddings,
                **sampling_kwargs,
            ).clone()

            # Generate
            input_pos = torch.tensor([T], device=prompt.device, dtype=torch.int)
            generated_tokens, _ = self.decode_n_tokens(
                next_token.view(1, -1),
                input_pos,
                max_new_tokens - 1,
                stop_first_eos,
                **sampling_kwargs,
            )

            # Fill an empty tensor with the generated tokens
            T_new = T + len(generated_tokens) + 1
            empty = torch.empty(T_new, dtype=prompt.dtype, device=prompt.device)
            empty[:T] = prompt
            seq = empty
            seq[T] = next_token
            seq[T + 1 :] = torch.cat(generated_tokens)

        # print(f"Generated sequence vlm:", self.tokenizer.decode(seq[self.text_model.config.n_image_tokens :].tolist()))
        if only_new_tokens:
            seq = seq[T:]
        elif remove_image_tokens:
            seq = seq[self.text_model.config.vision_args.n_image_tokens :]

        if do_decode:
            seq = self.tokenizer.decode(seq.tolist())
            return seq
        else:
            return seq


if __name__ == "__main__":
    torch.manual_seed(45)
    # model = GemmaVLMBase(model_name="hf-paligemma-3b-mix-224", compile=False, quant="none")
    model = VLMBase(
        model_name="llama3-llava-next-8b-hf",
        compile=False,
        quant="none",
        dtype=torch.bfloat16,
        device=torch.device("cuda"),
        # kv_cache_dtype=torch.float8_e5m2
    )
    # images = ["tennis_player.jpeg", "tennis_player.jpeg"]
    # prompts = ["Describe this image", "Describe this image"]
    # res = model.batch_generate(
    #     prompts,
    #     images,
    #     None,
    #     200,
    #     only_new_tokens=False,
    #     do_decode=True,
    #     temperature=0.8,
    #     top_k=200,
    #     only_first_image_features=True,
    # )
    # for x in res[0]:
    #     print("-------")
    #     print(x)

    # for n_batch_size in [16, 32, 64, 128]:
    #     print(f"Batch size: {n_batch_size}")
    #     model.benchmark_batch_tok_per_sec(n_batch_size, 200, 5, temperature=0.8, top_k=200)
    #     print("-------")

    # model.benchmark_batch_tok_per_sec(32, 200, 1, temperature=0.8, top_k=200)

    prompt = "Give me a very descriptive caption of this image.\n"
    image = "tennis_player.jpeg"
    print(
        model.generate(
            prompt,
            image,
            200,
            only_new_tokens=True,
            remove_image_tokens=False,
            only_first_image_features=True,
            temperature=0.8,
            top_k=200,
        )[0]
    )
    # model.benchmark_tok_per_s(prompt, image, 200, 5, temperature=0.8, top_k=200)
