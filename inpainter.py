# there is two stage one is inpainting and the other is lightning. 
# all the function are availabe you just need to run them.  
from my_nodes import *
from ip_adapter import IPAdapterUnifiedLoader
from modules.ip_Adapter.IP_Adapter import IPAdapterAdvanced
from comfy_extras.nodes_differential_diffusion import DifferentialDiffusion


def return_impainted():
   # stage 1 i.e inpainting 
   ref_image_path = "/Users/nitishupasi/Quicksnap/actions/comfy_ui/proj1/Data/ref_image.jpeg"
   prod_image_path = "/Users/nitishupasi/Quicksnap/actions/comfy_ui/proj1/Data/prod_img.jpeg"

   main_ckpt_path = "/Users/nitishupasi/Quicksnap/actions/comfy_ui/weights/juggernautXL_v9Rdphoto2Lightning.safetensors"
   depth_anything_ckpt_path = "/Users/nitishupasi/Quicksnap/actions/comfy_ui/weights/depth_anything_vitl14.pth"
   control_net_ckpt_path = "/Users/nitishupasi/Quicksnap/actions/comfy_ui/weights/control-lora-depth-rank256.safetensors"
   clip_vision_ckpt_path ="/Users/nitishupasi/Quicksnap/actions/comfy_ui/weights/clip_vision_model.safetensors"
   # initialise all the nodes
   image_loader = LoadImage()
   ckptloader = CheckpointLoaderSimple()
   clip_tokenizer = CLIPTextEncode()
   load_control_net = ControlNetLoader()
   apply_control_net = ControlNetApplyAdvanced()
   image_resizer = ImageResize()
   depth_anything= Depth_Anything_Preprocessor()
   get_bg_session = RemBGSession()
   # bg_remover = ImageRemoveBackground()
   mask_inverter = InvertMask()
   mask_grower = GrowMask()
   ip_adapt_loader = IPAdapterUnifiedLoader()
   prep_for_clip_vision = PrepImageForClipVision()
   ksampler = KSampler()
   inpaint_conditoning = InpaintModelConditioning()
   clip_vision_loader = CLIPVisionLoader()
   adv_ipadapter = IPAdapterAdvanced()
   diff_difffusion = DifferentialDiffusion()
   vae_decoder = VAEDecode()


   #loading image and prompts
   reference_image, _ = image_loader.load_image(ref_image_path)
   print("Processed Image:", reference_image.shape)
   product_image, _ = image_loader.load_image(prod_image_path)

   pos_prompt = "hello"
   neg_prompt = "bye"


   # get_model,clip,vae and prompts
   model_patcher,clip,vae = ckptloader.load_checkpoint(main_ckpt_path)
   pos_prompt_token = clip_tokenizer.encode(clip,pos_prompt)
   neg_prompt_token = clip_tokenizer.encode(clip,neg_prompt)


   # get depth maps for control_net
   resized_prod,_,_=image_resizer.execute(product_image,
                                    1024,
                                    0
                                 ) 

   # depth_image =  depth_anything.execute(resized_prod,
   #                               depth_anything_ckpt_path,
   #                               1024
   #                               )


   #get_mask
   remBG_session = get_bg_session.execute("u2net: general purpose","CPU")
   # _,mask = ImageRemoveBackground.execute(rembg_session= remBG_session,image=resized_prod)
   _,mask = ImageRemoveBackground_execute(rembg_session= remBG_session[0],image=resized_prod)
   inverted_mask = mask_inverter.invert(mask)
   growed_mask = mask_grower.expand_mask(inverted_mask[0],True,4)

   #controlnet
   control_net =load_control_net.load_controlnet(control_net_ckpt_path)

   pos_controlnet_token, neg_controlnet_token  = apply_control_net.apply_controlnet(pos_prompt_token[0],
                                                                        neg_prompt_token[0],
                                                                        control_net[0],
                                                                        mask,
                                                                        1,
                                                                        0,
                                                                        1)

   #IPAdapter
   prep_output = prep_for_clip_vision.prep_image(reference_image)
   ip_model,ip_dictinfo =ip_adapt_loader.load_models(model_patcher,"PLUS")
   clip_vision= clip_vision_loader.load_clip(clip_vision_ckpt_path)

   out,_ = adv_ipadapter.apply_ipadapter(model=ip_model,
                           ipadapter=ip_dictinfo,
                           image= prep_output[0],
                           clip_vision= clip_vision[0],
                           weight = 0.5,
                           weight_type= "linear",
                           combine_embeds = "concat",
                           start_at= 0,
                           end_at = 1,
                           embeds_scaling = 'V only'
   )

   dif_out=diff_difffusion.apply(out)

   ###### get inpaint conditioning
   pos,neg,latent = inpaint_conditoning.encode(pos_controlnet_token,
                                             neg_controlnet_token,
                                             resized_prod,
                                             vae,
                                             growed_mask[0])

   ####### run inppaint
   inpainted_sample  = ksampler.sample(dif_out[0],
                              1,
                              7,
                              2,
                              "dpmpp_sde",
                              "karras",
                              pos,
                              neg,
                              latent
                              )

   inpainted_image=vae_decoder.decode(inpainted_sample,vae)

   return inpainted_image , product_image