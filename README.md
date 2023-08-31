# Image Generation using Stable Diffusion 
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1nea9S_3uMRLkMDjfaLNv5fzHSUHSZovu?usp=sharing)

This command line application provides a convenient interface for generating images via artificial intelligence. It harnesses the power of Stable Diffusion, a leading AI system for text-to-image generation, in an easy-to-use package.

The application can be installed on a computer and invoked from the command line using the `generate` command. It accepts a variety of inputs including text prompts, random seeds, batch sizes, paths to control images, control image types, flags for custom styles, and more. These allow the user to tailor the image generation process.

Under the hood, the application runs the text-to-image diffusion process using Stable Diffusion 1.5. This produces high-quality images from the text prompts. The application also utilizes advanced techniques like Self-Attention Guidance and ControlNets to further refine the output.

The interface makes it simple to generate images either using the base Stable Diffusion model or with an original artistic style (Kerala Murals).

![compare_results](https://github.com/sushantmenon1/generate/assets/74258021/f80eccf1-1a11-4a59-89c3-c8e1ded85400)

## Installation 

Installing `generate` is simply
```bash
pip install git+https://github.com/sushantmenon1/generate.git
```
## Quickstart
```bash
generate --prompt="a village by the sea" 
```
or alternatively you can pass a prompt.txt file for muiltiple prompts
```bash
export PROMPT_DIR="path-to-text-file"

generate --prompt=$PROMPT_DIR --seed=1337
```
All the images will be saved in a created "Results" folder in the current working directory.

### Creating Multiple Images per prompt
You can generate multiple images for each prompt using the following command
```bash
export SEED_PATH="path-to-seed-file"

generate --prompt=$PROMPT_DIR -n=4 --seed=$SEED_PATH
```
#### Note: If you are generating multiple images a "seed.txt" file is needed. If `None` passed seeds will be automatically generated and saved in the "Results" folder for reproducibility.

## Available Parameters

|   | parameter                 | Description | 
|---|---------------------------|-----------------------|
|   | -p/--prompt | Prompts for generating Images. Can be a string or a text file.|
|   | -s/--seed  | Setting seeds for reproducibilty. If None generated_seeds.txt will be generated in the Results Folder. Can be an integer or a text file. Number of seeds must match the -n/--num_images parameter|
|   | -n/--num_images    | Number of Images to generate per prompt| 
|   | -i/--init_image    | Image path for ContolNet. Used if ControlNet is used|
|   | -c/--controlnet | Type of ControlNet to use. Choices are Canny and MiDaS|
|   | --style   | If 1, a trained style model (Kerala Murals) will be used for generating Images| 
|   | --sag_scale    | Self Attention Guidance (ranges between 0 and 1)| 

## Advanced Examples
### Style
For this project, I chose to train a new artistic style based on Kerala Mural paintings. As someone bought up in Kerala, India this vibrant painting style has always been close to my heart. Kerala Murals are known for their bright colors, ornate motifs, and epic mythological themes. They adorn temples and palaces across Kerala. To capture this aesthetic, I curated a dataset of Kerala Mural images showing figures, scenes, and decorative elements. I used this to fine-tune Stable Diffusion 1.5, developing a style capable of mimicking the colors, patterns, and composition of Kerala Murals. The result is an AI model imbued with the unique visual heritage of my home.


| ![sd_1 5](https://github.com/sushantmenon1/generate/assets/74258021/5d49afa3-064c-409e-8652-603932f02627) | ![styled_image](https://github.com/sushantmenon1/generate/assets/74258021/da8dbb25-082a-4e89-9625-4a899f327cdf) |
|:----------------------:|:----------------------:|
|      SD 1.5     |      Style     |

***Both these images were generate using the same prompt "a village by the sea" with the same seed (1337)***

Here is how you can Implement the Kerala Mural Painting Style Model.
```bash
generate --prompt=$PROMPT_DIR -n=4 --style=1
```

### Control Net
For enhanced control over the image generation process, I utilized two different ControlNets - Canny and MiDaS. 

Canny applies edge detection to locate and sharpen edges in the output image. This helps create well-defined outlines and crisp detailing. 

MiDaS stands for Monocular Depth Estimation. It adds a sense of depth and perspective to the generated image by estimating depth from a single image. MiDaS was trained on depth maps calculated from stereo images and can reproduce this effect for a given input image. 

Using these two ControlNets in conjunction allows for images with sharp edges and realistic depth. The Canny edges prevent blurring while the MiDaS depth introduces complex perspective and 3D appearance. 

Here is how you can use the ControlNet
```bash
export IMAGE_PATH="path-to-image"

generate --prompt=$PROMPT_DIR -n=4 --controlnet="Canny" --init_image=IMAGE_PATH
```
Alternatively you can also use the MiDaS ControlNet
```bash
export IMAGE_PATH="path-to-image"

generate --prompt=$PROMPT_DIR -n=4 --controlnet="MiDaS" --init_image=IMAGE_PATH
```
#### Note: Make sure to pass the directory of an Image in --init_image parameter for the ControlNet to use

### ControlNet with Style Model
You can also use the controlnet with the Style Model using this code
```bash
export IMAGE_PATH="path-to-image"

generate --prompt=$PROMPT_DIR -n=4 --controlnet="MiDaS" --init_image=IMAGE_PATH --style=1
```

### Self Attention Guidance

Self-Attention Guidance (SAG) is a technique to improve image generation in diffusion models like Stable Diffusion. It works by guiding the model's attention to focus on certain regions during image synthesis.

In the standard diffusion process, each step adds some noise to obscure the image. SAG modifies this by injecting additional input images that highlight desired areas like edges or boundaries. As the model steps through diffusion, it is conditioned on these input images to pay more attention to those salient regions.

This helps generate sharper, more coherent images. The model can use the attention guidance signals to retain important image features during the noisy diffusion process. Areas outlined by the input images will be less distorted and blurred.

Implementing SAG requires training the model to take in both the text prompt and attention guidance images at each diffusion step. At inference time, appropriate guidance images that match the prompt can make the output imagery more detailed.

Here is how you can implement SAG
```bash
generate --prompt=$PROMPT_DIR -n=4 --sag_scale=0.75
```
## Data
I collected Kerala Mural images from the net to fine-tune the base Stable Diffusion Model. 
I chose to train a Kerala mural painting style for this image generation project for a few key reasons:
- I have a personal connection to Kerala being originally from there, so I wanted to incorporate this cultural heritage into the AI model. Kerala murals are a unique artform from my home state.
- The vivid colors, intricate patterns, and flowing forms of Kerala mural art provide great visual material for developing a distinctive aesthetic. I felt this would translate well to a learned artistic style.
- There are not many examples I could find of AI models trained explicitly on Kerala mural art. So this presented an opportunity to expand the diversity of artistic styles represented in generative systems.
- Kerala mural paintings are not as universally known compared to Western art styles. I hoped training an AI style transfer on them would help highlight and preserve elements of this important cultural tradition through an emergent technology.

## Acknowledgements
- Diffusion Models from HuggingFace
- Vermillio - This application was built as a task for Vermillio
