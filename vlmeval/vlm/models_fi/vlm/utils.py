import numpy as np
from typing import Tuple
import random
import glob


def to_rgb(img):
    if img.shape[0] == 1:
        img = img.repeat(3, 1, 1)
    return img


def generate_batch_of_vlm_prompt(batch_size: int, path_to_images: str = None) -> Tuple:
    if path_to_images:
        image_files = glob.glob(
            path_to_images + "/*.jpg"
        )  # assuming images are in jpg format
        images = random.sample(image_files, batch_size)
    else:
        images = ["tennis_player.jpeg"] * batch_size
    prompts = [
        "Caption this image",
        "Descrive this image",
        "What do you see in this image",
    ]
    prompts = random.choices(prompts, k=batch_size)
    return images, prompts
