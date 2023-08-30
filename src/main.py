import argparse

from pathlib import Path
from .pipeline import Pipeline
from .utils import read_from_text_file


def generate():
    parser = argparse.ArgumentParser(description='A utility package to generate images from texts')
    
    parser.add_argument('-p', '--prompt', type=str, required=True, help='Text prompt')
    parser.add_argument('-s', '--seed', type=str, default=None, help='Seed for randomness')
    parser.add_argument('-n', '--num_images', type=int, default=1, help='Number of images')
    parser.add_argument('-i', '--init_image', type=str, default=None, help='Initial image for controlnet')
    parser.add_argument('-c', '--controlnet', type=str, default=None, choices=['Canny', 'HED', 'MiDaS', 'OpenPose'], help='ControlNet type')
    parser.add_argument('--style', type=int, default=0, help='Use custom style model')
    parser.add_argument('--sag_scale', type=float, default=0, help='SAG scale factor')
    
    args = parser.parse_args()

    # get prompt
    if Path(args.prompt).is_file():
        args.prompt = read_from_text_file(args.prompt)
    else:
        args.prompt = [args.prompt]

    # get seed
    if args.seed and Path(args.seed).is_file():
        args.seed = read_from_text_file(args.seed)
    else:
        args.seed = [args.seed] if args.seed else None

    # style
    if args.style:
        args.prompt = [f"kerala mural painting of {prompt}" for prompt in args.prompt]

    # validating inputs
    if args.seed and len(args.seed) != args.num_images:
        raise ValueError("Length of seeds and num_images should match")
    if args.controlnet and not args.init_image:
        raise ValueError("ControlNet used but no control image passed")
    if args.controlnet and args.sag_scale:
        raise ValueError("ControlNet support not implemented with SAG")

    # create output directory
    output_dir = Path('Results')
    output_dir.mkdir(parents=True, exist_ok=True) 

    # generate
    pipe = Pipeline(args)
    generated_images = pipe.generate()

    # save images to disk
    for i, (prompt, images) in enumerate(generated_images.items()):
        prompt_dir = output_dir.joinpath(f"{i}_{prompt}")
        prompt_dir.mkdir(parents=True, exist_ok=True)
        for j, image in enumerate(images):
            image.save(prompt_dir.joinpath(f"{j}.png"))

if __name__ == '__main__':
    generate()
