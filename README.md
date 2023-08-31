# Image Generation using Stable Diffusion

## Abstract

This command line application provides a convenient interface for generating images via artificial intelligence. It harnesses the power of Stable Diffusion, a leading AI system for text-to-image generation, in an easy-to-use package.

The application can be installed on a computer and invoked from the command line using the "generate" command. It accepts a variety of inputs including text prompts, random seeds, batch sizes, paths to control images, control image types, flags for custom styles, and more. These allow the user to tailor the image generation process.

Under the hood, the application runs the text-to-image diffusion process using Stable Diffusion 1.5. This produces high-quality images from the text prompts. The application also utilizes advanced techniques like Self-Attention Guidance and ControlNets to further refine the output.

In addition, the application can train new artistic styles that can be applied to generate images with custom aesthetics. This allows users to develop their own styles suited to their needs. The interface makes it simple to generate images either using the base Stable Diffusion model or with an original artistic style.

## Installation 

Clone this repository: git clone https://github.com/user/image-generation

## TO-DOs

1. Complete README.md (Installation, Usage, Arguments, Advanced Usage, Acknowledgements)
2. Polish -help in `main.py`
3. Polish comments (give credit to borrowed code / library)
4. Add doctrings to functions
5. Polish and commit notebook
6. Test release on macos
7. Test release on Windows
8. Model improvement
9. Submit
