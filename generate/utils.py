import subprocess
import cv2
import numpy as np
from PIL import Image

from pathlib import Path
from diffusers.utils import load_image


def read_from_text_file(path):
    with open(path, "r") as f:
        text = f.read().split('\n')
    return text

def write_to_text_file(path, data):
    # create file path 
    filepath = Path(path)
    # open file for writing 
    with filepath.open(mode='w') as f:
        for i in data:
            f.write(str(i)+'\n')

def preprocess_canny(path):
    image = load_image(path)
    image = np.array(image)
    low_threshold = 100
    high_threshold = 200
    image = cv2.Canny(image, low_threshold, high_threshold)
    image = image[:, :, None]
    image = np.concatenate([image, image, image], axis=2)
    canny_image = Image.fromarray(image)
    canny_image = canny_image.resize((512,512))
    return canny_image

def download_models(model_dir):
    # download weights
    subprocess.run(['curl', 
                    'https://storage.googleapis.com/playground-sushant-eefk/custom%20model/pytorch_custom_diffusion_weights.bin', 
                    '-o', 
                    model_dir.joinpath('pytorch_custom_diffusion_weights.bin')])

    # download encodings
    subprocess.run(['curl', 
                    'https://storage.googleapis.com/playground-sushant-eefk/custom%20model/kerala.bin', 
                    '-o', 
                    model_dir.joinpath('kerala.bin')])
