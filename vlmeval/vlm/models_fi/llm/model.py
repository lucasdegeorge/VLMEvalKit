import torch
import time
from pathlib import Path
from typing import Tuple, List, Optional
import torch._dynamo.config
import torch._inductor.config
import numpy as np
from typing import Tuple, Union
import contextlib
import os
import sys
from trycast import isassignable

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from llm.architecture.base import Transformer
from llm.tokenizer import get_tokenizer
from llm.utils import (
    sample,
    generate_batch_of_questions,
    remove_all_backward_hooks,
    generate_sentence,
)
from llm.architecture.gemma import TransformerForGemma
from llm.chat import Dialogue

from llm.tensor_parallel import (
    maybe_init_dist,
    initialize_model_parallel,
    apply_tp,
    get_model_parallel_rank,
    get_data_parallel_rank,
    get_data_parallel_world_size,
)

from jinja2.exceptions import TemplateError
from jinja2.sandbox import ImmutableSandboxedEnvironment

torch._inductor.config.coordinate_descent_tuning = True
torch._inductor.config.triton.unique_kernel_names = True
torch._inductor.config.fx_graph_cache = True  # Experimental feature to reduce compilation times, will be on by default in future


class LLMBase:
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

        self.model = self.load_model(quant=quant, use_tp=use_tp)

        self.tokenizer = get_tokenizer(model_name, self.checkpoint_path, None)
        self.eos_id = self.tokenizer.eos_id()
        self.pad_id = self.tokenizer.pad_id()

        ## Check and remove this part !!
        # For Jinja template: a dict mapping special tokens (`cls_token`, `unk_token`, etc.) to their values (`'<unk>'`, `'<cls>'`, etc.).
        # Special tokens can be added to this dict using the `add_special_token` method.
        self.special_tokens_map = {
            "bos_token": "<|begin_of_text|>",
            "eos_token": "<|end_of_text|>",
        }
        # self.special_tokens_map = {'bos_token': "<s>", 'eos_token': "</s>"}

    def forward(self, x: torch.Tensor):
        return self.model(x)

    def load_model(self, quant: str = "none", use_tp: bool = False):
        tick = time.time()
        with torch.device("meta"):
            model = Transformer.from_name(self.model_name)
            model.vocab_parallel = False  ## TODO: check if this is needed
            self.checkpoint_path = model.config.checkpoint_path
            if isinstance(self.checkpoint_path, str):
                self.checkpoint_path = Path(self.checkpoint_path)

        if quant == "none":
            self.model_path = self.checkpoint_path / "model.pth"
            assert self.model_path.is_file(), str(self.model_path)

        elif quant == "int8":
            self.model_path = self.checkpoint_path / "model_int8.pth"
            assert self.model_path.is_file(), str(self.model_path)
            from llm.quantization import WeightOnlyInt8QuantHandler

            simple_quantizer = WeightOnlyInt8QuantHandler(model)
            model = simple_quantizer.convert_for_runtime()

        elif quant == "int4":
            self.model_path = self.checkpoint_path / "model_int4.g32.pth"
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

    def tokenize(
        self, string: str, bos: bool = True, eos: bool = False
    ) -> torch.Tensor:
        assert isinstance(string, str), "Input must be a string."
        tokens = self.tokenizer.encode(string, bos=bos, eos=eos)
        return torch.tensor(tokens, dtype=torch.int, device=self.device)

    def batch_tokenize(
        self, strings: list, bos: List[bool] = None, eos: List[bool] = None
    ) -> torch.Tensor:
        assert isinstance(strings, list), "Input must be a list of strings."
        assert all(
            isinstance(s, str) for s in strings
        ), "Input must be a list of strings."
        return self.tokenizer.batch_encode(
            strings, bos=bos, eos=eos, device=self.device
        )

    def decode_one_token(
        self,
        x: torch.Tensor,
        input_pos: torch.Tensor,
        left_pad_mask_pos: torch.Tensor,
        **sampling_kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # input_pos: [B, 1]
        assert input_pos.shape[-1] == 1
        logits = self.model(
            idx=x, input_pos=input_pos, embeds=None, left_pad_mask_pos=left_pad_mask_pos
        )
        return sample(logits, self.model.vocab_parallel, **sampling_kwargs)

    def decode_n_tokens(
        self,
        cur_token: torch.Tensor,
        input_pos: torch.Tensor,
        num_new_tokens: int,
        stop_first_eos: bool = True,
        **sampling_kwargs,
    ):
        new_tokens, new_probs = [], []
        for i in range(num_new_tokens):
            # with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH):
            with torch.backends.cuda.sdp_kernel(
                enable_flash=False, enable_mem_efficient=False, enable_math=True
            ):
                next_token, next_prob = self.decode_one_token(
                    cur_token, input_pos, None, **sampling_kwargs
                )
                input_pos += 1
                new_tokens.append(next_token.clone())
                new_probs.append(next_prob.clone())
                cur_token = next_token.view(1, -1)
                try:
                    if stop_first_eos and next_token in self.tokenizer.stop_tokens:
                        # print(f"Found EOS token at position {i}")
                        break
                except RuntimeError as e:
                    print(new_tokens[-1])
                    print(next_token.clone())
                    print(self.tokenizer.stop_tokens)
                    raise e
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
        """embeds: useles for LLM (here to match the signature of the VLM method)"""
        # input_pos: [B, S]
        assert embeds is None
        logits = self.model(x, input_pos, embeds, left_pad_mask_pos)
        return sample(logits, self.model.vocab_parallel, **sampling_kwargs)

    @torch.no_grad()
    def generate(
        self,
        prompt: Union[torch.Tensor, str, Dialogue],
        max_new_tokens: int,
        do_decode: bool = True,
        stop_first_eos: bool = True,
        only_new_tokens: bool = False,
        clean_dialogue: bool = False,
        **sampling_kwargs,
    ) -> Union[torch.Tensor, str, Dialogue]:

        if self.compile:
            self.decode_one_token = torch.compile(
                self.decode_one_token, mode="reduce-overhead", fullgraph=True
            )

        if clean_dialogue and not (isassignable(prompt, Dialogue)):
            raise ValueError("clean_dialogue can only be used with Dialogue inputs.")

        if clean_dialogue and not (do_decode):
            raise ValueError("clean_dialogue can only be used with do_decode=True.")

        if isinstance(prompt, str):
            prompt = self.tokenize(prompt, bos=True, eos=False)

        if isassignable(prompt, Dialogue):
            dialogue = prompt
            prompt = self.apply_chat_template(prompt, return_tensor=True)

        if self.use_tp:
            print("use_tp is useless for non-batched inference. Ignoring it.")

        prompt = prompt.to(device=self.device)

        # Generation
        T = prompt.size(0)
        T_max = T + max_new_tokens
        max_seq_length = min(T_max, self.model.config.block_size)

        self.model.setup_caches(
            max_batch_size=1,
            max_seq_length=max_seq_length,
            prompt_size=T,
            device=self.device,
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
                embeds=None,
                left_pad_mask_pos=None,
                **sampling_kwargs,
            )

        # Generate
        input_pos = torch.tensor([T], device=prompt.device, dtype=torch.int)
        generated_tokens, _, nb_generated_tokens = self.decode_n_tokens(
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

        if only_new_tokens:
            seq = seq[T:]

        if do_decode:
            seq = self.tokenizer.decode(seq.tolist())

        if clean_dialogue:
            ## TODO: provide a way to clean the dialogue for each model
            seq = seq.split("<|end_header_id|>")[-1].split("<|eot_id|>")[0]
            dialogue.append({"role": "llm", "content": seq})
            return dialogue

        return seq, nb_generated_tokens

    @torch.no_grad()
    def batch_generate(
        self,
        prompt: Union[torch.Tensor, List[str], Dialogue],
        left_pad_mask_pos: torch.Tensor,
        max_new_tokens: int,
        do_decode: bool = True,
        stop_first_eos: bool = True,
        only_new_tokens: bool = False,
        clean_dialogue: bool = False,
        **sampling_kwargs,
    ) -> Union[torch.Tensor, str, Dialogue]:

        if self.compile:
            self.decode_one_token = torch.compile(
                self.decode_one_token, mode="reduce-overhead", fullgraph=True
            )

        # if clean_dialogue and not (isassignable(prompt, Dialogue)):
        #     raise ValueError("clean_dialogue can only be used with Dialogue inputs.")

        # if clean_dialogue and not (do_decode):
        #     raise ValueError("clean_dialogue can only be used with do_decode=True.")

        if isinstance(prompt, list) and all(isinstance(s, str) for s in prompt):
            prompt, left_pad_mask_pos = self.batch_tokenize(
                prompt, bos=[True] * len(prompt), eos=[False] * len(prompt)
            )

        # if isassignable(prompt, Dialogue):
        #     dialogue = prompt
        #     prompt = self.apply_chat_template(prompt, return_tensor=True)

        if self.use_tp:
            torch.distributed.barrier()
            dp_rank = get_data_parallel_rank()
            tp_rank = get_model_parallel_rank()
            dp_size = get_data_parallel_world_size()

        # if self.compile:
        #     remove_all_backward_hooks(self.model)

        prompt = prompt.to(device=self.device)
        left_pad_mask_pos = left_pad_mask_pos.to(device=self.device)

        # Generation
        B = prompt.size(0)
        T = prompt.size(1)
        T_max = T + max_new_tokens
        max_seq_length = min(T_max, self.model.config.block_size)

        self.model.setup_caches(
            max_batch_size=B,
            max_seq_length=max_seq_length,
            prompt_size=T,
            device=self.device,
            kv_dtype=self.kv_cache_dtype,
        )
        input_pos = torch.arange(0, T, device=prompt.device)

        # with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH):
        with torch.backends.cuda.sdp_kernel(
            enable_flash=False, enable_mem_efficient=False, enable_math=True
        ):
            next_token, _ = self.prefill(
                prompt,
                input_pos=input_pos,
                embeds=None,
                left_pad_mask_pos=left_pad_mask_pos,
                **sampling_kwargs,
            )

        # Generate
        input_pos = torch.tensor([T], device=self.device, dtype=torch.int)
        generated_tokens, _, nb_generated_tokens = self.batch_decode_n_tokens(
            next_token.view(B, -1),
            input_pos,
            left_pad_mask_pos,
            max_new_tokens - 1,
            stop_first_eos,
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
            seq = self.tokenizer.batch_decode(seq.tolist())

        # if clean_dialogue:
        #     seq = seq.split("<|end_header_id|>")[-1].split("<|eot_id|>")[0]
        #     dialogue.append({"role": "llm", "content": seq})
        #     return dialogue

        return seq, nb_generated_tokens

    def apply_chat_template(self, conversation, **kwargs) -> Union[str, torch.Tensor]:
        """
        This method should be overridden by subclasses.
        Must return a string or a tensor containing the chat tokens.
        """
        raise NotImplementedError("This method should be overridden by subclasses.")

    @property
    def chat_template(self):
        """This method should be overridden by subclasses.
        It is used by SentencePiece based models to format a dialogue into a chat prompt.

        For TikToks based models:
        It should return a ChatFormat object that can be used to encode a dialogue into a chat prompt.

        For SentencePiece based models:
        It should return a Jinja template string that can be used to render a chat conversation.
        See the corresponding template in the Hugging Face API
        """
        raise NotImplementedError("This method should be overridden by subclasses.")

    def _compile_jinja_template(self, chat_template):
        def raise_exception(message):
            raise TemplateError(message)

        jinja_env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True)
        jinja_env.policies["json.dumps_kwargs"]["ensure_ascii"] = False
        jinja_env.globals["raise_exception"] = raise_exception
        return jinja_env.from_string(chat_template)

    def add_special_token(self, tokens: dict):
        for key, value in tokens.items():
            self.special_tokens_map[key] = value

    def benchmark_tok_per_s(
        self, prompt: str, max_new_tokens: int, nb_samples: int = 5, **sampling_kwargs
    ) -> float:
        tokens_per_s = list()
        for i in range(-1, nb_samples):
            t0 = time.perf_counter()
            encoded = self.tokenize(prompt, bos=True)
            output, nb_generated_tokens = self.generate(
                encoded,
                max_new_tokens,
                do_decode=False,
                stop_first_eos=False,
                **sampling_kwargs,
            )
            if i == -1:
                print(f"Compilation time: {time.perf_counter() - t0:.2f} seconds")
                continue
            else:
                # print(self.tokenizer.decode(output.tolist()))
                # print("-----------------------")
                tokens_generated = output.size(0) - encoded.size(0)
                tokens_per_s.append(tokens_generated / (time.perf_counter() - t0))
        print(f"Average tokens per second: {np.mean(tokens_per_s):.2f}")
        return np.mean(tokens_per_s)

    def benchmark_batch_tok_per_sec(
        self,
        batch_size: int,
        prompt_length: int,
        max_new_tokens: int,
        nb_samples: int = 5,
        **sampling_kwargs,
    ) -> float:
        tokens_per_s = list()
        gen_time = list()
        for i in range(-1, nb_samples):
            # batch = generate_batch_of_questions(batch_size, prompt_length)
            batch = [generate_sentence(prompt_length) for _ in range(batch_size)]
            t0 = time.perf_counter()
            encoded, left_pad_mask_pos = self.batch_tokenize(
                batch, bos=[True] * batch_size, eos=[False] * batch_size
            )
            output, nb_generated_tokens = self.batch_generate(
                prompt=encoded,
                left_pad_mask_pos=left_pad_mask_pos,
                max_new_tokens=max_new_tokens,
                do_decode=False,
                stop_first_eos=False,
                **sampling_kwargs,
            )
            if i == -1:
                print(f"Compilation time: {time.perf_counter() - t0:.2f} seconds")
                continue
            else:
                # print(self.tokenizer.decode(output.tolist()))
                # print("-----------------------")
                gen_time.append(time.perf_counter() - t0)
                tokens_generated_sec = nb_generated_tokens * output.size(0)
                tokens_per_s.append(tokens_generated_sec / (time.perf_counter() - t0))
        print(f"Average tokens per second: {np.mean(tokens_per_s):.2f}")
        print(f"Average generation time: {np.mean(gen_time):.2f}")
        return np.mean(tokens_per_s)


class GemmaBase(LLMBase):
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
        super().__init__(
            model_name, compile, quant, use_tp, dtype, kv_cache_dtype, device
        )

    def load_model(self, quant: str = "none", use_tp: bool = False):
        tick = time.time()
        with torch.device("meta"):

            ## New line for Gemma models
            model = TransformerForGemma.from_name(self.model_name)

            self.checkpoint_path = model.config.checkpoint_path
            if isinstance(self.checkpoint_path, str):
                self.checkpoint_path = Path(self.checkpoint_path)

        if quant == "none":
            self.model_path = self.checkpoint_path / "model.pth"
            assert self.model_path.is_file(), str(self.model_path)

        elif quant == "int8":
            self.model_path = self.checkpoint_path / "model_int8.pth"
            assert self.model_path.is_file(), str(self.model_path)
            from llm.quantization import WeightOnlyInt8QuantHandler

            simple_quantizer = WeightOnlyInt8QuantHandler(model)
            model = simple_quantizer.convert_for_runtime()

        elif quant == "int4":
            self.model_path = self.checkpoint_path / "model_int4.g32.pth"
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


if __name__ == "__main__":
    torch.manual_seed(45)
    model = LLMBase(
        model_name="Meta-Llama-3-8B-Instruct",
        compile=False,
        quant="none",
        dtype=torch.bfloat16,
        device="cuda:0",
    )
    # model = GemmaBase(
    #     model_name="gemma-2b-it",
    #     compile=True,
    #     quant="none",
    #     dtype=torch.bfloat16,
    #     device="cuda:0",
    # )
    prompt = ["Where Bill Gates is born?", "What is the capital of France?"]
    res = model.batch_generate(
        prompt, None, 200, only_new_tokens=False, temperature=0.8, top_k=200
    )
    for x in res[0]:
        print("-------")
        print(x)

    # import matplotlib.pyplot as plt

    # batch_sizes = [16, 32, 64, 128, 256, 512]
    # lengths = [10, 50, 100, 250, 500, 1000, 2000, 4000]
    # speeds = []

    # for length in lengths:
    #     model = LLMBase(
    #         model_name="Meta-Llama-3-8B-Instruct",
    #         compile=True,
    #         quant="none",
    #         dtype=torch.bfloat16,
    #         device="cuda:0",
    #     )
    #     speed_batch = []
    #     for batch_size in batch_sizes:
    #         print(f"Batch size: {batch_size}")
    #         print(f"Length: {length}")
    #         speed = model.benchmark_batch_tok_per_sec(
    #             batch_size, length, 100, nb_samples=2
    #         )
    #         speed_batch.append(speed)
    #         print("------------------------------------------------")
    #     speeds.append(speed_batch)
    #     del model
    #     torch.cuda.empty_cache()

    # # Plotting
    # array = np.array(speeds).transpose()

    # for i, batch_size in enumerate(batch_sizes):
    #     plt.plot(lengths, speeds[i], label=f"Batch size: {batch_size}")

    # plt.xlabel("Length")
    # plt.ylabel("Speed (tokens per second)")
    # plt.legend()
    # plt.show()

    # np.save("speeds.npy", array)

    # prompt = "Hello, tell me a joke"
    # print(
    #     model.generate(prompt, 200, only_new_tokens=False, temperature=0.8, top_k=200)[
    #         0
    #     ]
    # )
    # model.benchmark_tok_per_s(prompt, 200, nb_samples=5)
