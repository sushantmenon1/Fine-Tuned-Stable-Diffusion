import torch

from pathlib import Path
from .utils import preprocess_canny, write_to_text_file
# all the pipeline wrappers are taken from the HuggingFace's Diffusers library, see https://github.com/huggingface/diffusers
from diffusers import StableDiffusionPipeline, StableDiffusionSAGPipeline, ControlNetModel, StableDiffusionControlNetPipeline, logging
from diffusers.utils import load_image


class Pipeline:
    def __init__(self, args):
        self.args = args
        self.custom_model_path = Path(__file__).parent.joinpath("model")
        self.stable_diffusion_checkpoint = "runwayml/stable-diffusion-v1-5"
        self.controlnet_checkpoints = {'Canny': 'lllyasviel/sd-controlnet-canny', 
                                       'MiDaS': 'lllyasviel/sd-controlnet-depth'}        
        self.width = 512
        self.height = 512

        # configure device
        if torch.backends.mps.is_available():
          self.device = torch.device("mps")
        elif torch.cuda.is_available():
          self.device = torch.device("cuda")
        else:
          self.device = torch.device("cpu")

        # set pipeline
        self.torch_dtype = torch.float32 
        # if str(self.device) == 'mps' else torch.float16
        self.pipe = self.load_controlnet() if self.args.controlnet else self.load_stable_diffusion()

    def generate(self):
        latents = self.generate_latents()
        generated_images = {}
        for prompt in self.args.prompt:
            kwargs = {'prompt': [prompt] * self.args.num_images, 'latents': latents}
            if self.args.controlnet:
                kwargs['image'] = self.load_init_image()
            if self.args.sag_scale:
                kwargs['sag_scale'] = self.args.sag_scale

            #with torch.autocast(str(self.device)):
            images = self.pipe(**kwargs)["images"]
            generated_images[prompt] = images

        return generated_images

    def load_controlnet(self):
        self.controlnet_name = self.controlnet_checkpoints[self.args.controlnet]
        controlnet = ControlNetModel.from_pretrained(self.controlnet_name)
        pipe = StableDiffusionControlNetPipeline.from_pretrained(self.stable_diffusion_checkpoint, 
                                                                 torch_dtype=self.torch_dtype,
                                                                 controlnet=controlnet)

        if self.args.style:
            pipe.unet.load_attn_procs(self.custom_model_path, 
                                      weight_name="pytorch_custom_diffusion_weights.bin")
            pipe.load_textual_inversion(self.custom_model_path, 
                                        weight_name="kerala.bin")

        # send the pipeline to the appropriate device
        pipe = pipe.to(self.device)

        return pipe

    def load_stable_diffusion(self):
        if self.args.sag_scale:
            pipe = StableDiffusionSAGPipeline.from_pretrained(self.stable_diffusion_checkpoint, 
                                                              torch_dtype=self.torch_dtype, 
                                                              use_safetensors=True)
        else:
             pipe = StableDiffusionPipeline.from_pretrained(self.stable_diffusion_checkpoint, 
                                                            torch_dtype=self.torch_dtype, 
                                                            use_safetensors=True)
        
        if self.args.style:
            pipe.unet.load_attn_procs(self.custom_model_path, 
                                      weight_name="pytorch_custom_diffusion_weights.bin")
            pipe.load_textual_inversion(self.custom_model_path, 
                                        weight_name="kerala.bin")

        # send the pipeline to the appropriate device
        pipe = pipe.to(self.device)

        return pipe

    def load_init_image(self):
        if self.controlnet_name == self.controlnet_checkpoints['Canny']:
            image = preprocess_canny(self.args.init_image)
        else:
            image = load_image(self.args.init_image).resize((512,512))
            
        return image
        
    # https://github.com/pcuenca/diffusers-examples/blob/main/notebooks/stable-diffusion-seeds.ipynb
    def generate_latents(self):
        generator = torch.Generator(device=self.device)
        if not self.args.seed:
            self.args.seed = [generator.seed() for i in range(self.args.num_images)]
            write_to_text_file("Results/generated_seeds.txt", data=self.args.seed)

        latents = None
        for i in range(self.args.num_images):
            # get a new random seed, store it and use it as the generator state
            generator = generator.manual_seed(int(self.args.seed[i]))
            image_latents = torch.randn((1, self.pipe.unet.config.in_channels, self.height // 8, self.width // 8),
                                        generator=generator,
                                        device=self.device)
            latents = image_latents if latents is None else torch.cat((latents, image_latents))
        
        return latents

            

        

