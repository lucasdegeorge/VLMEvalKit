import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional, Dict, Tuple
from collections import OrderedDict
import random
from torch.distributed import _functional_collectives as funcol
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from vlm.image_encoder.pixtral_utils import _reshape_for_broadcast


class ClassInstantier(OrderedDict):
    def __getitem__(self, key):
        content = super().__getitem__(key)
        cls, kwargs = content if isinstance(content, tuple) else (content, {})
        return cls(**kwargs)


activation_classes = {
    "leaky_relu": nn.LeakyReLU,
    "relu": nn.ReLU,
    "sigmoid": nn.Sigmoid,
    "silu": nn.SiLU,
    "tanh": nn.Tanh,
    "gelu": (nn.GELU, {"approximate": "tanh"}),
}
activation_functions = ClassInstantier(activation_classes)


def find_multiple(n: int, k: int) -> int:
    if n % k == 0:
        return n
    return n + k - (n % k)


def precompute_freqs_cis(
    seq_len: int,
    n_elem: int,
    base: int = 10000,
    dtype: torch.dtype = torch.bfloat16,
    device: torch.device = torch.device("cuda"),
) -> Tensor:
    freqs = 1.0 / (
        base ** (torch.arange(0, n_elem, 2)[: (n_elem // 2)].float() / n_elem)
    )
    t = torch.arange(seq_len, device=freqs.device)
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    cache = torch.stack([freqs_cis.real, freqs_cis.imag], dim=-1)
    return cache.to(dtype=dtype, device=device)


def fi_rotary_emb(x: Tensor, freqs_cis: Tensor) -> Tensor:
    xshaped = x.float().reshape(*x.shape[:-1], -1, 2)
    freqs_cis = freqs_cis.view(1, xshaped.size(1), 1, xshaped.size(3), 2)
    x_out2 = torch.stack(
        [
            xshaped[..., 0] * freqs_cis[..., 0] - xshaped[..., 1] * freqs_cis[..., 1],
            xshaped[..., 1] * freqs_cis[..., 0] + xshaped[..., 0] * freqs_cis[..., 1],
        ],
        -1,
    )
    x_out2 = x_out2.flatten(3)
    return x_out2.type_as(x)


def hf_rotary_emb(x: Tensor, cos, sin, unsqueeze_dim=1) -> Tensor:
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)

    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    rotated_x = torch.cat((-x2, x1), dim=-1)

    x_embed = (x * cos) + (rotated_x * sin)
    return x_embed


# def vit_apply_rotary_emb(
#     x: torch.Tensor,
#     freqs_cis: torch.Tensor,
# ) -> Tuple[torch.Tensor, torch.Tensor]:
#     x_ = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
#     assert freqs_cis.dtype == torch.complex64
#     freqs_cis = _reshape_for_broadcast(freqs_cis, x_)
#     x_out = torch.view_as_real(x_ * freqs_cis).flatten(3)
#     return x_out.type_as(x)


def sample(logits, vocab_parallel, temperature: float = 1, top_k: Optional[int] = None):
    if logits.shape[0] == 1:
        probs = logits_to_probs(logits[0, -1], temperature, top_k)
        idx_next = multinomial_sample_one_no_sync(probs)
        return idx_next, probs
    else:
        with torch.autocast(device_type="cuda", enabled=False):
            from llm.tensor_parallel import get_model_parallel_group

            logits = logits[:, -1].float()
            if vocab_parallel:
                logits = funcol.all_gather_tensor(
                    logits, gather_dim=-1, group=get_model_parallel_group()
                )
            probs = logits_to_probs(logits, temperature, top_k)
            idx_next = multinomial_sample_one_no_sync(probs)
        return idx_next, probs


def logits_to_probs(logits, temperature: float = 0.8, top_k: int = 200):
    logits = logits / max(temperature, 1e-5)

    if top_k is not None:
        v, _ = torch.topk(logits, min(top_k, logits.size(-1)), dim=-1)
        pivot = v.select(-1, -1).unsqueeze(-1)
        logits = torch.where(logits < pivot, -float("Inf"), logits)
    probs = torch.nn.functional.softmax(logits, dim=-1)
    return probs


def multinomial_sample_one_no_sync(
    probs_sort,
):  # Does multinomial sampling without a cuda synchronization
    q = torch.empty_like(probs_sort).exponential_(1)
    return torch.argmax(probs_sort / q, dim=-1, keepdim=True).to(dtype=torch.int)


def device_sync(device):
    if "cuda" in device:
        torch.cuda.synchronize(device)
    elif ("cpu" in device) or ("mps" in device):
        pass
    else:
        print(f"device={device} is not yet suppported")


def remove_all_backward_hooks(model: torch.nn.Module) -> Dict[str, OrderedDict]:
    all_backward_hooks = {}
    for name, module in model.named_modules():
        all_backward_hooks[name] = module._backward_hooks
        module._backward_hooks = OrderedDict()
    return all_backward_hooks


subjects = [
    "The cat",
    "A dog",
    "The quick brown fox",
    "A lazy dog",
    "The happy bird",
    "The curious monkey",
    "A sleepy bear",
]
verbs = ["jumps", "runs", "sleeps", "eats", "flies", "climbs", "swims", "dances"]
objects = [
    "over the lazy dog",
    "in the park",
    "on the mat",
    "the food",
    "in the sky",
    "up the tree",
    "in the river",
    "on the stage",
]
question_starters = ["Does", "Will", "Can", "Should", "Would"]


def generate_random_question():
    starter = random.choice(question_starters)
    subject = random.choice(subjects)
    verb = random.choice(verbs)
    obj = random.choice(objects)
    return f"{starter} {subject} {verb} {obj}?"


def generate_sentence(length):
    words = ["the", "quick", "brown", "fox", "jumps", "over", "the", "lazy", "dog"]
    sentence = " ".join(random.choice(words) for _ in range(length))
    # sentence = sentence.capitalize() + '.'
    return sentence


def generate_batch_of_questions(batch_size):
    return [generate_random_question() for _ in range(batch_size)]
