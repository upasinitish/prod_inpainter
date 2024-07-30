from my_nodes import *
from modules.Video_Matting.comfyui_vidmatt.briaai_rembg import BriaaiRembg

def IC_light(inpainted_img,prod_img):
    image_resizer = ImageResize()
    brai_matting = BriaaiRembg()
    ckptloader = CheckpointLoaderSimple()
    clip_tokenizer = CLIPTextEncode()
    vae_encode = VAEEncode()
    vae_decode = VAEDecode()

    


    
