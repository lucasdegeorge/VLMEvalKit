from trycast import isassignable
import torch
from typing import Union
from model import LLMBase
from chat import ChatFormat, Dialogue


class Llama3(LLMBase):
    def __init__(
        self,
        model_name: str = "Meta-Llama-3-8B-Instruct",
        compile: bool = True,
        quant: str = "int8",
    ):
        super().__init__(model_name, compile, quant)
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


if __name__ == "__main__":
    torch.manual_seed(45)
    model = Llama3(model_name="Meta-Llama-3-8B-Instruct", quant="none")
    # prompt = [
    #     {"role": "user", "content": "We are playing a game. I have selected an image. By asking me questions, you objective is to write the best captions possible to recover the original image from your caption. First, ask me one (and only one) direct questions about the image and wait for my answer. Ensure questions are closely tied to the image content. Emphasize questions that focus on intricate details, like recognizing objects, pinpointing positions, identifying colors, counting quantities, feeling moods, and more. Do NOT imagine, invent or infer anything. Ask me one question (and give only the question without other text). A short caption of the image is: One man playing tennis"},
    #     {"role": "assistant", "content": "How the man is dressed?"},
    #     {"role": "user", "content": "The man is dressed with a white t-shirt and black shorts. Aske me another question."},
    # ]
    prompt = [
        {
            "role": "user",
            "content": "We are playing a game. I have selected an image. By asking me questions, you objective is to write the best captions possible to recover the original image from your caption. First, ask me one (and only one) direct questions about the image and wait for my answer. Ensure questions are closely tied to the image content. Emphasize questions that focus on intricate details, like recognizing objects, pinpointing positions, identifying colors, counting quantities, feeling moods, and more. Do NOT imagine, invent or infer anything. Ask me one question (and give only the question without other text). A short caption of the image is: A bell hanging from an outdoor structure. ",
        },
        {"role": "assistant", "content": "Is the bell hanging from a church steeple?"},
        {
            "role": "user",
            "content": "Yes, the bell is hung to a church steeple, which is an outdoor structure typically found at churches. Ask me another question. sfvzefffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffvvvvvvvvvvvvvvvvvvefvzevzebzbzbzbezbze zef g zg z e zt z rg sg gr rg rzg zrg hzrgh erg sefhg fh  gsgh g g sg sg Yes, the bell is hung to a church steeple, which is an outdoor structure typically found at churches. Ask me another question. sfvzefffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffvvvvvvvvvvvvvvvvvvefvzevzebzbzbzbezbze zef g zg z e zt z rg sg gr rg rzg zrg hzrgh erg sefhg fh  gsgh g g sg sgYes, the bell is hung to a church steeple, which is an outdoor structure typically found at churches. Ask me another question. sfvzefffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffvvvvvvvvvvvvvvvvvvefvzevzebzbzbzbezbze zef g zg z e zt z rg sg gr rg rzg zrg hzrgh erg sefhg fh  gsgh g g sg sgYes, the bell is hung to a church steeple, which is an outdoor structure typically found at churches. Ask me another question. sfvzefffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffvvvvvvvvvvvvvvvvvvefvzevzebzbzbzbezbze zef g zg z e zt z rg sg gr rg rzg zrg hzrgh erg sefhg fh  gsgh g g sg sgYes, the bell is hung to a church steeple, which is an outdoor structure typically found at churches. Ask me another question. sfvzefffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffvvvvvvvvvvvvvvvvvvefvzevzebzbzbzbezbze zef g zg z e zt z rg sg gr rg rzg zrg hzrgh erg sefhg fh  gsgh g g sg sgYes, the bell is hung to a church steeple, which is an outdoor structure typically found at churches. Ask me another question. sfvzefffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffvvvvvvvvvvvvvvvvvvefvzevzebzbzbzbezbze zef g zg z e zt z rg sg gr rg rzg zrg hzrgh erg sefhg fh  gsgh g g sg sgYes, the bell is hung to a church steeple, which is an outdoor structure typically found at churches. Ask me another question. sfvzefffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffvvvvvvvvvvvvvvvvvvefvzevzebzbzbzbezbze zef g zg z e zt z rg sg gr rg rzg zrg hzrgh erg sefhg fh  gsgh g g sg sgYes, the bell is hung to a church steeple, which is an outdoor structure typically found at churches. Ask me another question. sfvzefffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffvvvvvvvvvvvvvvvvvvefvzevzebzbzbzbezbze zef g zg z e zt z rg sg gr rg rzg zrg hzrgh erg sefhg fh  gsgh g g sg sgYes, the bell is hung to a church steeple, which is an outdoor structure typically found at churches. Ask me another question. sfvzefffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffvvvvvvvvvvvvvvvvvvefvzevzebzbzbzbezbze zef g zg z e zt z rg sg gr rg rzg zrg hzrgh erg sefhg fh  gsgh g g sg sgYes, the bell is hung to a church steeple, which is an outdoor structure typically found at churches. Ask me another question. sfvzefffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffvvvvvvvvvvvvvvvvvvefvzevzebzbzbzbezbze zef g zg z e zt z rg sg gr rg rzg zrg hzrgh erg sefhg fh  gsgh g g sg sg  ",
        },
    ]
    # prompt = [
    #     {"role": "user", "content": "Hello! Tell me a joke."},
    # ]
    print(model.generate(prompt, 50, clean_dialogue=False, temperature=0.8, top_k=200))
    # model.benchmark_tok_per_s(prompt, 200, nb_samples=5)
