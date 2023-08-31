import logging
import argparse

from pathlib import Path
from .pipeline import Pipeline
from .utils import read_from_text_file, download_models

# set up logger
logging.basicConfig(format='%(asctime)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S')
logging.root.setLevel(logging.INFO)

def generate():
    parser = argparse.ArgumentParser(description='A utility package to generate images from texts')
    parser.add_argument('-p', '--prompt', type=str, required=True, help='Prompts for generating Images. Can be a string or a text file')
    parser.add_argument('-s', '--seed', type=str, default=None, help='Setting seeds for reproducibilty. If None generated_seeds.txt will be generated in the Results Folder. Can be an integer or a text file. Number of seeds must match the -n/--num_images parameter')
    parser.add_argument('-n', '--num_images', type=int, default=1, help='Number of Images to generate per prompt')
    parser.add_argument('-i', '--init_image', type=str, default=None, help='Image path for ContolNet. Used if ControlNet is used')
    parser.add_argument('-c', '--controlnet', type=str, default=None, choices=['Canny', 'MiDaS'], help='Type of ControlNet to use. Choices are Canny and MiDaS')
    parser.add_argument('--style', type=int, default=0, help='If 1, a trained style model (Kerala Murals) will be used for generating Images')
    parser.add_argument('--sag_scale', type=float, default=0, help='Self Attention Guidance (ranges between 0 and 1)')
    args = parser.parse_args()

    # download models if they don't exist
    model_dir = Path(__file__).parent.joinpath("model")
    if not model_dir.exists():
        logging.info('Pulling model weights')
        model_dir.mkdir()
        download_models(model_dir)
    
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
    logging.info('Generating images')
    generated_images = pipe.generate()

    # save images to disk
    logging.info(f'Saving generated images to {output_dir}')
    for i, (prompt, images) in enumerate(generated_images.items()):
        prompt_dir = output_dir.joinpath(f"{i}_{prompt}")
        prompt_dir.mkdir(parents=True, exist_ok=True)
        for j, image in enumerate(images):
            image.save(prompt_dir.joinpath(f"{j}.png"))

if __name__ == '__main__':
    generate()
