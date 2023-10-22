import os
import random
import sys
from typing import Sequence, Mapping, Any, Union
import torch
from datetime import datetime


def get_value_at_index(obj: Union[Sequence, Mapping], index: int) -> Any:
    """Returns the value at the given index of a sequence or mapping.

    If the object is a sequence (like list or string), returns the value at the given index.
    If the object is a mapping (like a dictionary), returns the value at the index-th key.

    Some return a dictionary, in these cases, we look for the "results" key

    Args:
        obj (Union[Sequence, Mapping]): The object to retrieve the value from.
        index (int): The index of the value to retrieve.

    Returns:
        Any: The value at the given index.

    Raises:
        IndexError: If the index is out of bounds for the object and the object is not a mapping.
    """
    try:
        return obj[index]
    except KeyError:
        return obj["result"][index]


def find_path(name: str, path: str = None) -> str:
    """
    Recursively looks at parent folders starting from the given path until it finds the given name.
    Returns the path as a Path object if found, or None otherwise.
    """
    # If no path is given, use the current working directory
    if path is None:
        path = os.getcwd()

    # Check if the current directory contains the name
    if name in os.listdir(path):
        path_name = os.path.join(path, name)
        print(f"{name} found: {path_name}")
        return path_name

    # Get the parent directory
    parent_directory = os.path.dirname(path)

    # If the parent directory is the same as the current directory, we've reached the root and stop the search
    if parent_directory == path:
        return None

    # Recursively call the function with the parent directory
    return find_path(name, parent_directory)


def add_comfyui_directory_to_sys_path() -> None:
    """
    Add 'ComfyUI' to the sys.path
    """
    comfyui_path = find_path("ComfyUI")
    if comfyui_path is not None and os.path.isdir(comfyui_path):
        sys.path.append(comfyui_path)
        print(f"'{comfyui_path}' added to sys.path")


def add_extra_model_paths() -> None:
    """
    Parse the optional extra_model_paths.yaml file and add the parsed paths to the sys.path.
    """
    from main import load_extra_path_config

    extra_model_paths = find_path("extra_model_paths.yaml")

    if extra_model_paths is not None:
        load_extra_path_config(extra_model_paths)
    else:
        print("Could not find the extra_model_paths config file.")


add_comfyui_directory_to_sys_path()
add_extra_model_paths()


def import_custom_nodes() -> None:
    """Find all custom nodes in the custom_nodes folder and add those node objects to NODE_CLASS_MAPPINGS

    This function sets up a new asyncio event loop, initializes the PromptServer,
    creates a PromptQueue, and initializes the custom nodes.
    """
    import asyncio
    import execution
    from nodes import init_custom_nodes
    import server

    # Creating a new event loop and setting it as the default loop
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    # Creating an instance of PromptServer with the loop
    server_instance = server.PromptServer(loop)
    execution.PromptQueue(server_instance)

    # Initializing custom nodes
    init_custom_nodes()


from nodes import (
    ConditioningCombine,
    ConditioningConcat,
    SetLatentNoiseMask,
    EmptyLatentImage,
    VAEEncode,
    ConditioningAverage,
    VAEDecode,
    LoadImage,
    ImageScale,
    NODE_CLASS_MAPPINGS,
)


def main():
    import_custom_nodes()
    with torch.inference_mode():
        seargeprompttext = NODE_CLASS_MAPPINGS["SeargePromptText"]()
        seargeprompttext_5 = seargeprompttext.get_value(prompt="")

        seargeprompttext_6 = seargeprompttext.get_value(prompt="")

        seargeprompttext_7 = seargeprompttext.get_value(
            prompt="clothing, text, blurry, logo, watermark, signature, cropped, out of frame, worst quality, low quality, jpeg artifacts, poorly lit, overexposed, underexposed, glitch, error, out of focus, (semi-realistic, cgi, 3d, render, sketch, cartoon, drawing, digital art, anime, manga:1.3), amateur, (poorly drawn hands, poorly drawn face:1.2), deformed iris, deformed pupils, morbid, duplicate, mutilated, extra fingers, mutated hands, poorly drawn eyes, mutation, deformed, bad anatomy, bad proportions, extra limbs, cloned face, disfigured, gross proportions, malformed limbs, missing arms, missing legs, extra arms, extra legs, fused fingers, too many fingers, long neck, incoherent,"
        )

        seargeprompttext_8 = seargeprompttext.get_value(prompt="")

        seargeinput7 = NODE_CLASS_MAPPINGS["SeargeInput7"]()
        seargeinput7_150 = seargeinput7.mux(
            lora_strength=0.1, operation_mode="inpainting", prompt_style="simple"
        )

        seargeinput6 = NODE_CLASS_MAPPINGS["SeargeInput6"]()
        seargeinput6_272 = seargeinput6.mux(
            hires_fix="disabled",
            hrf_steps=25,
            hrf_denoise=0.25,
            hrf_upscale_factor=1.5,
            hrf_intensity="hard",
            hrf_seed_offset="distinct",
            hrf_smoothness=0.49999999999999994,
            inputs=get_value_at_index(seargeinput7_150, 0),
        )

        seargeinput5 = NODE_CLASS_MAPPINGS["SeargeInput5"]()
        seargeinput5_143 = seargeinput5.mux(
            base_conditioning_scale=2,
            refiner_conditioning_scale=2,
            style_prompt_power=0.333,
            negative_style_power=0.667,
            style_template="none",
            inputs=get_value_at_index(seargeinput6_272, 0),
        )

        seargeinput2 = NODE_CLASS_MAPPINGS["SeargeInput2"]()

        seargeinput3 = NODE_CLASS_MAPPINGS["SeargeInput3"]()

        seargeinput1 = NODE_CLASS_MAPPINGS["SeargeInput1"]()
        loadimage = LoadImage()
        invertmask = NODE_CLASS_MAPPINGS["InvertMask"]()
        seargestylepreprocessor = NODE_CLASS_MAPPINGS["SeargeStylePreprocessor"]()
        seargeparameterprocessor = NODE_CLASS_MAPPINGS["SeargeParameterProcessor"]()
        seargeoutput7 = NODE_CLASS_MAPPINGS["SeargeOutput7"]()
        seargeoutput2 = NODE_CLASS_MAPPINGS["SeargeOutput2"]()
        seargeoutput1 = NODE_CLASS_MAPPINGS["SeargeOutput1"]()
        seargeoutput5 = NODE_CLASS_MAPPINGS["SeargeOutput5"]()
        seargeintegerscaler = NODE_CLASS_MAPPINGS["SeargeIntegerScaler"]()
        seargeinput4 = NODE_CLASS_MAPPINGS["SeargeInput4"]()
        seargeinput4_154 = seargeinput4.mux(
            base_model="Stable-diffusion/sd_xl_base_1.0.safetensors",
            refiner_model="Stable-diffusion/sd_xl_refiner_1.0.safetensors",
            vae_model="sdxl_vae.safetensors",
            main_upscale_model="RealESRGAN_x2plus.pth",
            support_upscale_model="RealESRGAN_x2plus.pth",
            lora_model="sd_xl_offset_example-lora_1.0.safetensors",
        )
        seargeoutput4 = NODE_CLASS_MAPPINGS["SeargeOutput4"]()
        seargeoutput4_349 = seargeoutput4.demux(
            model_names=get_value_at_index(seargeinput4_154, 0)
        )
        seargeoutput3 = NODE_CLASS_MAPPINGS["SeargeOutput3"]()
        seargegenerated1 = NODE_CLASS_MAPPINGS["SeargeGenerated1"]()
        conditioningconcat = ConditioningConcat()
        conditioningaverage = ConditioningAverage()
        conditioningcombine = ConditioningCombine()
        seargeconditioningmuxer5 = NODE_CLASS_MAPPINGS["SeargeConditioningMuxer5"]()
        seargeconditioningmuxer2 = NODE_CLASS_MAPPINGS["SeargeConditioningMuxer2"]()
        emptylatentimage = EmptyLatentImage()
        masktoimage = NODE_CLASS_MAPPINGS["MaskToImage"]()
        imagetomask = NODE_CLASS_MAPPINGS["ImageToMask"]()
        feathermask = NODE_CLASS_MAPPINGS["FeatherMask"]()
        setlatentnoisemask = SetLatentNoiseMask()
        seargelatentmuxer3 = NODE_CLASS_MAPPINGS["SeargeLatentMuxer3"]()
        seargesdxlsampler2 = NODE_CLASS_MAPPINGS["SeargeSDXLSampler2"]()
        vaedecode = VAEDecode()
        seargeimagesave = NODE_CLASS_MAPPINGS["SeargeImageSave"]()
        seargecheckpointloader = NODE_CLASS_MAPPINGS["SeargeCheckpointLoader"]()
        seargeloraloader = NODE_CLASS_MAPPINGS["SeargeLoraLoader"]()
        seargesdxlpromptencoder = NODE_CLASS_MAPPINGS["SeargeSDXLPromptEncoder"]()
        seargesdxlbasepromptencoder = NODE_CLASS_MAPPINGS[
            "SeargeSDXLBasePromptEncoder"
        ]()
        seargesdxlrefinerpromptencoder = NODE_CLASS_MAPPINGS[
            "SeargeSDXLRefinerPromptEncoder"
        ]()
        imagescale = ImageScale()
        seargevaeloader = NODE_CLASS_MAPPINGS["SeargeVAELoader"]()
        vaeencode = VAEEncode()
        seargeintegerconstant = NODE_CLASS_MAPPINGS["SeargeIntegerConstant"]()
        fixed_seed = 15570221102832527854
        seargeinput2_133 = seargeinput2.mux(
            seed=fixed_seed,
            image_width=768,
            image_height=1344,
            steps=40,
            cfg=7,
            sampler_name="euler_ancestral",
            scheduler="karras",
            save_image="enabled",
            save_directory="output folder",
            inputs=get_value_at_index(seargeinput5_143, 0),
        )
        seargeinput3_136 = seargeinput3.mux(
            base_ratio=0.7999999999999998,
            refiner_strength=0.7499999999999998,
            refiner_intensity="hard",
            precondition_steps=0,
            batch_size=1,
            upscale_resolution_factor=1,
            save_upscaled_image="enabled",
            denoise=0.666,
            inputs=get_value_at_index(seargeinput2_133, 0),
        )
        seargevaeloader_351 = seargevaeloader.load_vae(
            vae_name=get_value_at_index(seargeoutput4_349, 3)
        )
        seargecheckpointloader_350 = seargecheckpointloader.load_checkpoint(
            ckpt_name=get_value_at_index(seargeoutput4_349, 1)
        )

        run = datetime.now().strftime("%d.%m.%Y-%H:%M:%S")
        embedding = "christphgrmmr-step00003050"
        embedding_name = "christphgrmmr"

        seargeinput1_270 = seargeinput1.mux(
            main_prompt="(embedding:" + embedding + ")\nModel photoshoot for clothing retailer website of nice looking " + embedding_name + " with a clear background, (Documentary Photography), beautiful, best quality, high definition, studio lighting, white background, gradient background (masterpiece), (extremely intricate), deviantart hd, artstation hd, detailed face, award-winning photography, margins, detailed face, intricate, high detail, sharp focus, dramatic, award winning matte cinematic lighting volumetrics dtx",
            secondary_prompt=get_value_at_index(seargeprompttext_5, 0),
            style_prompt=get_value_at_index(seargeprompttext_6, 0),
            negative_prompt=get_value_at_index(seargeprompttext_7, 0),
            negative_style=get_value_at_index(seargeprompttext_8, 0),
            inputs=get_value_at_index(seargeinput3_136, 0),
            image=None,
            mask=None,
        )

        seargeparameterprocessor_131 = seargeparameterprocessor.process(
            inputs=get_value_at_index(seargeinput1_270, 0)
        )

        seargeoutput7_152 = seargeoutput7.demux(
            parameters=get_value_at_index(seargeparameterprocessor_131, 0)
        )
        seargeloraloader_353 = seargeloraloader.load_lora(
            strength_model=get_value_at_index(seargeoutput7_152, 1),
            strength_clip=get_value_at_index(seargeoutput7_152, 1),
            model=get_value_at_index(seargecheckpointloader_350, 0),
            clip=get_value_at_index(seargecheckpointloader_350, 1),
            lora_name=get_value_at_index(seargeoutput4_349, 6),
        )
        seargeoutput3_369 = seargeoutput3.demux(
            parameters=get_value_at_index(seargeoutput7_152, 0)
        )
        seargeoutput2_359 = seargeoutput2.demux(
            parameters=get_value_at_index(seargeoutput3_369, 0)
        )
        seargecheckpointloader_355 = seargecheckpointloader.load_checkpoint(
            ckpt_name=get_value_at_index(seargeoutput4_349, 2)
        )
        jpg_paths = []
        for root, _, files in os.walk(
                "/home/luca/virtual-tryon-diffusion/datasets/deep-fashion-test/img_highres/MEN"):
            for f in files:
                if f.lower().endswith('.jpg'):
                    jpg_paths.append(os.path.join(root, f))
        print("found", len(jpg_paths), "images")

        for path in jpg_paths:
            result_path = path.split(".")[0] + "_"+embedding+".png"

            loadimage_1 = loadimage.load_image(
                image=path)
            loadimage_mask = loadimage.load_image(
                image=path.split(".")[0] + "_mask_sam.png")
            mask = imagetomask.image_to_mask(
                channel="green", image=get_value_at_index(loadimage_mask, 0)
            )

            seargeinput1_270 = seargeinput1.mux(
                main_prompt="(embedding:" + embedding + ")\nModel photoshoot for clothing retailer website of nice looking " + embedding_name + " with a clear background, (Documentary Photography), beautiful, best quality, high definition, studio lighting, white background, gradient background (masterpiece), (extremely intricate), deviantart hd, artstation hd, detailed face, award-winning photography, margins, detailed face, intricate, high detail, sharp focus, dramatic, award winning matte cinematic lighting volumetrics dtx",
                secondary_prompt=get_value_at_index(seargeprompttext_5, 0),
                style_prompt=get_value_at_index(seargeprompttext_6, 0),
                negative_prompt=get_value_at_index(seargeprompttext_7, 0),
                negative_style=get_value_at_index(seargeprompttext_8, 0),
                inputs=get_value_at_index(seargeinput3_136, 0),
                image=get_value_at_index(loadimage_1, 0),
                mask=get_value_at_index(mask, 0),
            )
            seargeparameterprocessor_131 = seargeparameterprocessor.process(
                inputs=get_value_at_index(seargeinput1_270, 0)
            )
            seargeoutput7_152 = seargeoutput7.demux(
                parameters=get_value_at_index(seargeparameterprocessor_131, 0)
            )
            seargeoutput3_369 = seargeoutput3.demux(
                parameters=get_value_at_index(seargeoutput7_152, 0)
            )
            seargeoutput2_359 = seargeoutput2.demux(
                parameters=get_value_at_index(seargeoutput3_369, 0)
            )


            seargeimagesave.save_images(
                filename_prefix="transform_all/" + run + "/" + result_path + "_mask.png",
                images=get_value_at_index(loadimage_mask, 0),
                state=get_value_at_index(seargeoutput2_359, 8),
                save_to=get_value_at_index(seargeoutput2_359, 9),
            )


            seargeoutput2_370 = seargeoutput2.demux(
                parameters=get_value_at_index(seargeoutput7_152, 0)
            )

            seargeoutput1_153 = seargeoutput1.demux(
                parameters=get_value_at_index(seargeoutput2_370, 0)
            )

            seargeoutput5_161 = seargeoutput5.demux(
                parameters=get_value_at_index(seargeoutput1_153, 0)
            )

            seargeintegerscaler_51 = seargeintegerscaler.get_value(
                value=get_value_at_index(seargeoutput2_370, 2),
                factor=get_value_at_index(seargeoutput5_161, 1),
                multiple_of=8,
            )

            seargeintegerscaler_52 = seargeintegerscaler.get_value(
                value=get_value_at_index(seargeoutput2_370, 3),
                factor=get_value_at_index(seargeoutput5_161, 1),
                multiple_of=8,
            )

            seargeintegerscaler_54 = seargeintegerscaler.get_value(
                value=get_value_at_index(seargeoutput2_370, 2),
                factor=get_value_at_index(seargeoutput5_161, 2),
                multiple_of=8,
            )

            seargeintegerscaler_53 = seargeintegerscaler.get_value(
                value=get_value_at_index(seargeoutput2_370, 3),
                factor=get_value_at_index(seargeoutput5_161, 2),
                multiple_of=8,
            )

            seargesdxlpromptencoder_40 = seargesdxlpromptencoder.encode(
                pos_g=get_value_at_index(seargeoutput1_153, 1),
                pos_l=get_value_at_index(seargeoutput1_153, 2),
                pos_r=get_value_at_index(seargeoutput1_153, 1),
                neg_g=get_value_at_index(seargeoutput1_153, 4),
                neg_l=get_value_at_index(seargeoutput1_153, 4),
                neg_r=get_value_at_index(seargeoutput1_153, 4),
                base_width=get_value_at_index(seargeintegerscaler_51, 0),
                base_height=get_value_at_index(seargeintegerscaler_52, 0),
                crop_w=0,
                crop_h=0,
                target_width=get_value_at_index(seargeoutput2_370, 2),
                target_height=get_value_at_index(seargeoutput2_370, 3),
                pos_ascore=6,
                neg_ascore=2.5,
                refiner_width=get_value_at_index(seargeintegerscaler_54, 0),
                refiner_height=get_value_at_index(seargeintegerscaler_53, 0),
                base_clip=get_value_at_index(seargeloraloader_353, 1),
                refiner_clip=get_value_at_index(seargecheckpointloader_355, 1),
            )

            seargesdxlbasepromptencoder_41 = seargesdxlbasepromptencoder.encode(
                pos_g=get_value_at_index(seargeoutput1_153, 3),
                pos_l=get_value_at_index(seargeoutput1_153, 3),
                neg_g=get_value_at_index(seargeoutput1_153, 5),
                neg_l=get_value_at_index(seargeoutput1_153, 5),
                base_width=get_value_at_index(seargeintegerscaler_51, 0),
                base_height=get_value_at_index(seargeintegerscaler_52, 0),
                crop_w=0,
                crop_h=0,
                target_width=get_value_at_index(seargeoutput2_370, 2),
                target_height=get_value_at_index(seargeoutput2_370, 3),
                base_clip=get_value_at_index(seargeloraloader_353, 1),
            )

            seargesdxlrefinerpromptencoder_42 = seargesdxlrefinerpromptencoder.encode(
                pos_r=get_value_at_index(seargeoutput1_153, 3),
                neg_r=get_value_at_index(seargeoutput1_153, 5),
                pos_ascore=6,
                neg_ascore=2.5,
                refiner_width=get_value_at_index(seargeintegerscaler_54, 0),
                refiner_height=get_value_at_index(seargeintegerscaler_53, 0),
                refiner_clip=get_value_at_index(seargecheckpointloader_355, 1),
            )

            seargeoutput2_290 = seargeoutput2.demux(
                parameters=get_value_at_index(seargeoutput7_152, 0)
            )

            seargeoutput1_293 = seargeoutput1.demux(
                parameters=get_value_at_index(seargeoutput2_290, 0)
            )
            imagescale_281 = imagescale.upscale(
                upscale_method="bicubic",
                width=get_value_at_index(seargeoutput2_290, 2),
                height=get_value_at_index(seargeoutput2_290, 3),
                crop="center",
                image=get_value_at_index(seargeoutput1_293, 6),
            )

            vaeencode_282 = vaeencode.encode(
                pixels=get_value_at_index(imagescale_281, 0),
                vae=get_value_at_index(seargevaeloader_351, 0),
            )

            seargeintegerconstant_458 = seargeintegerconstant.get_value(value=0)

            seargeoutput5_145 = seargeoutput5.demux(
                parameters=get_value_at_index(seargeoutput7_152, 0)
            )

            seargegenerated1_334 = seargegenerated1.demux(
                parameters=get_value_at_index(seargeoutput5_145, 0)
            )

            conditioningconcat_313 = conditioningconcat.concat(
                conditioning_to=get_value_at_index(seargesdxlpromptencoder_40, 0),
                conditioning_from=get_value_at_index(seargesdxlbasepromptencoder_41, 0),
            )

            conditioningconcat_321 = conditioningconcat.concat(
                conditioning_to=get_value_at_index(seargesdxlbasepromptencoder_41, 0),
                conditioning_from=get_value_at_index(seargesdxlpromptencoder_40, 0),
            )

            conditioningaverage_75 = conditioningaverage.addWeighted(
                conditioning_to_strength=get_value_at_index(seargeoutput5_145, 3),
                conditioning_to=get_value_at_index(seargesdxlbasepromptencoder_41, 0),
                conditioning_from=get_value_at_index(seargesdxlpromptencoder_40, 0),
            )

            conditioningcombine_55 = conditioningcombine.combine(
                conditioning_1=get_value_at_index(seargesdxlpromptencoder_40, 0),
                conditioning_2=get_value_at_index(seargesdxlbasepromptencoder_41, 0),
            )

            seargeconditioningmuxer5_337 = seargeconditioningmuxer5.mux(
                input_selector=get_value_at_index(seargegenerated1_334, 2),
                input0=get_value_at_index(seargesdxlpromptencoder_40, 0),
                input1=get_value_at_index(conditioningconcat_313, 0),
                input2=get_value_at_index(conditioningconcat_321, 0),
                input3=get_value_at_index(conditioningaverage_75, 0),
                input4=get_value_at_index(conditioningcombine_55, 0),
            )

            seargeconditioningmuxer5_376 = seargeconditioningmuxer5.mux(
                input_selector=get_value_at_index(seargegenerated1_334, 2),
                input0=get_value_at_index(conditioningconcat_313, 0),
                input1=get_value_at_index(conditioningconcat_321, 0),
                input2=get_value_at_index(seargesdxlbasepromptencoder_41, 0),
                input3=get_value_at_index(conditioningaverage_75, 0),
                input4=get_value_at_index(conditioningcombine_55, 0),
            )

            seargeconditioningmuxer2_439 = seargeconditioningmuxer2.mux(
                input_selector=get_value_at_index(seargegenerated1_334, 3),
                input0=get_value_at_index(seargeconditioningmuxer5_337, 0),
                input1=get_value_at_index(seargeconditioningmuxer5_376, 0),
            )

            conditioningconcat_318 = conditioningconcat.concat(
                conditioning_to=get_value_at_index(seargesdxlpromptencoder_40, 1),
                conditioning_from=get_value_at_index(seargesdxlbasepromptencoder_41, 1),
            )

            conditioningconcat_322 = conditioningconcat.concat(
                conditioning_to=get_value_at_index(seargesdxlbasepromptencoder_41, 1),
                conditioning_from=get_value_at_index(seargesdxlpromptencoder_40, 1),
            )

            conditioningaverage_76 = conditioningaverage.addWeighted(
                conditioning_to_strength=get_value_at_index(seargeoutput5_145, 4),
                conditioning_to=get_value_at_index(seargesdxlbasepromptencoder_41, 1),
                conditioning_from=get_value_at_index(seargesdxlpromptencoder_40, 1),
            )

            conditioningcombine_56 = conditioningcombine.combine(
                conditioning_1=get_value_at_index(seargesdxlpromptencoder_40, 1),
                conditioning_2=get_value_at_index(seargesdxlbasepromptencoder_41, 1),
            )

            seargeconditioningmuxer5_336 = seargeconditioningmuxer5.mux(
                input_selector=get_value_at_index(seargegenerated1_334, 2),
                input0=get_value_at_index(seargesdxlpromptencoder_40, 1),
                input1=get_value_at_index(conditioningconcat_318, 0),
                input2=get_value_at_index(conditioningconcat_322, 0),
                input3=get_value_at_index(conditioningaverage_76, 0),
                input4=get_value_at_index(conditioningcombine_56, 0),
            )

            seargeconditioningmuxer5_375 = seargeconditioningmuxer5.mux(
                input_selector=get_value_at_index(seargegenerated1_334, 2),
                input0=get_value_at_index(conditioningconcat_322, 0),
                input1=get_value_at_index(conditioningconcat_318, 0),
                input2=get_value_at_index(seargesdxlbasepromptencoder_41, 1),
                input3=get_value_at_index(conditioningcombine_56, 0),
                input4=get_value_at_index(conditioningaverage_76, 0),
            )

            seargeconditioningmuxer2_440 = seargeconditioningmuxer2.mux(
                input_selector=get_value_at_index(seargegenerated1_334, 3),
                input0=get_value_at_index(seargeconditioningmuxer5_336, 0),
                input1=get_value_at_index(seargeconditioningmuxer5_375, 0),
            )

            seargeconditioningmuxer2_454 = seargeconditioningmuxer2.mux(
                input_selector=get_value_at_index(seargeintegerconstant_458, 0),
                input0=get_value_at_index(seargeconditioningmuxer2_439, 0),
                input1=get_value_at_index(seargeconditioningmuxer2_440, 0),
            )

            seargeconditioningmuxer2_455 = seargeconditioningmuxer2.mux(
                input_selector=get_value_at_index(seargeintegerconstant_458, 0),
                input0=get_value_at_index(seargeconditioningmuxer2_440, 0),
                input1=get_value_at_index(seargeconditioningmuxer2_439, 0),
            )

            conditioningconcat_319 = conditioningconcat.concat(
                conditioning_to=get_value_at_index(seargesdxlpromptencoder_40, 2),
                conditioning_from=get_value_at_index(
                    seargesdxlrefinerpromptencoder_42, 0
                ),
            )

            conditioningconcat_323 = conditioningconcat.concat(
                conditioning_to=get_value_at_index(
                    seargesdxlrefinerpromptencoder_42, 0
                ),
                conditioning_from=get_value_at_index(seargesdxlpromptencoder_40, 2),
            )

            conditioningaverage_77 = conditioningaverage.addWeighted(
                conditioning_to_strength=get_value_at_index(seargeoutput5_145, 3),
                conditioning_to=get_value_at_index(
                    seargesdxlrefinerpromptencoder_42, 0
                ),
                conditioning_from=get_value_at_index(seargesdxlpromptencoder_40, 2),
            )

            conditioningcombine_57 = conditioningcombine.combine(
                conditioning_1=get_value_at_index(seargesdxlpromptencoder_40, 2),
                conditioning_2=get_value_at_index(seargesdxlrefinerpromptencoder_42, 0),
            )

            seargeconditioningmuxer5_338 = seargeconditioningmuxer5.mux(
                input_selector=get_value_at_index(seargegenerated1_334, 2),
                input0=get_value_at_index(seargesdxlpromptencoder_40, 2),
                input1=get_value_at_index(conditioningconcat_319, 0),
                input2=get_value_at_index(conditioningconcat_323, 0),
                input3=get_value_at_index(conditioningaverage_77, 0),
                input4=get_value_at_index(conditioningcombine_57, 0),
            )

            seargeconditioningmuxer5_377 = seargeconditioningmuxer5.mux(
                input_selector=get_value_at_index(seargegenerated1_334, 2),
                input0=get_value_at_index(conditioningconcat_319, 0),
                input1=get_value_at_index(conditioningconcat_323, 0),
                input2=get_value_at_index(seargesdxlrefinerpromptencoder_42, 0),
                input3=get_value_at_index(conditioningaverage_77, 0),
                input4=get_value_at_index(conditioningcombine_57, 0),
            )

            seargeconditioningmuxer2_441 = seargeconditioningmuxer2.mux(
                input_selector=get_value_at_index(seargegenerated1_334, 3),
                input0=get_value_at_index(seargeconditioningmuxer5_338, 0),
                input1=get_value_at_index(seargeconditioningmuxer5_377, 0),
            )

            conditioningconcat_320 = conditioningconcat.concat(
                conditioning_to=get_value_at_index(seargesdxlpromptencoder_40, 3),
                conditioning_from=get_value_at_index(
                    seargesdxlrefinerpromptencoder_42, 1
                ),
            )

            conditioningconcat_324 = conditioningconcat.concat(
                conditioning_to=get_value_at_index(
                    seargesdxlrefinerpromptencoder_42, 1
                ),
                conditioning_from=get_value_at_index(seargesdxlpromptencoder_40, 3),
            )

            conditioningaverage_78 = conditioningaverage.addWeighted(
                conditioning_to_strength=get_value_at_index(seargeoutput5_145, 4),
                conditioning_to=get_value_at_index(
                    seargesdxlrefinerpromptencoder_42, 1
                ),
                conditioning_from=get_value_at_index(seargesdxlpromptencoder_40, 3),
            )

            conditioningcombine_58 = conditioningcombine.combine(
                conditioning_1=get_value_at_index(seargesdxlpromptencoder_40, 3),
                conditioning_2=get_value_at_index(seargesdxlrefinerpromptencoder_42, 1),
            )

            seargeconditioningmuxer5_339 = seargeconditioningmuxer5.mux(
                input_selector=get_value_at_index(seargegenerated1_334, 2),
                input0=get_value_at_index(seargesdxlpromptencoder_40, 3),
                input1=get_value_at_index(conditioningconcat_320, 0),
                input2=get_value_at_index(conditioningconcat_324, 0),
                input3=get_value_at_index(conditioningaverage_78, 0),
                input4=get_value_at_index(conditioningcombine_58, 0),
            )

            seargeconditioningmuxer5_378 = seargeconditioningmuxer5.mux(
                input_selector=get_value_at_index(seargegenerated1_334, 2),
                input0=get_value_at_index(conditioningconcat_324, 0),
                input1=get_value_at_index(conditioningconcat_320, 0),
                input2=get_value_at_index(seargesdxlrefinerpromptencoder_42, 1),
                input3=get_value_at_index(conditioningcombine_58, 0),
                input4=get_value_at_index(conditioningaverage_78, 0),
            )

            seargeconditioningmuxer2_442 = seargeconditioningmuxer2.mux(
                input_selector=get_value_at_index(seargegenerated1_334, 3),
                input0=get_value_at_index(seargeconditioningmuxer5_339, 0),
                input1=get_value_at_index(seargeconditioningmuxer5_378, 0),
            )

            seargeconditioningmuxer2_456 = seargeconditioningmuxer2.mux(
                input_selector=get_value_at_index(seargeintegerconstant_458, 0),
                input0=get_value_at_index(seargeconditioningmuxer2_441, 0),
                input1=get_value_at_index(seargeconditioningmuxer2_442, 0),
            )

            seargeconditioningmuxer2_457 = seargeconditioningmuxer2.mux(
                input_selector=get_value_at_index(seargeintegerconstant_458, 0),
                input0=get_value_at_index(seargeconditioningmuxer2_442, 0),
                input1=get_value_at_index(seargeconditioningmuxer2_441, 0),
            )

            seargegenerated1_340 = seargegenerated1.demux(
                parameters=get_value_at_index(seargeoutput2_359, 0)
            )

            emptylatentimage_44 = emptylatentimage.generate(
                width=get_value_at_index(seargeoutput2_359, 2),
                height=get_value_at_index(seargeoutput2_359, 3),
                batch_size=get_value_at_index(seargeoutput3_369, 6),
            )

            masktoimage_286 = masktoimage.mask_to_image(
                mask=get_value_at_index(seargeoutput1_293, 7)
            )

            imagescale_284 = imagescale.upscale(
                upscale_method="bicubic",
                width=get_value_at_index(seargeoutput2_290, 2),
                height=get_value_at_index(seargeoutput2_290, 3),
                crop="center",
                image=get_value_at_index(masktoimage_286, 0),
            )

            imagetomask_285 = imagetomask.image_to_mask(
                channel="green", image=get_value_at_index(imagescale_284, 0)
            )

            feathermask_288 = feathermask.feather(
                left=4,
                top=4,
                right=4,
                bottom=4,
                mask=get_value_at_index(imagetomask_285, 0),
            )

            setlatentnoisemask_283 = setlatentnoisemask.set_mask(
                samples=get_value_at_index(vaeencode_282, 0),
                mask=get_value_at_index(feathermask_288, 0),
            )

            seargelatentmuxer3_335 = seargelatentmuxer3.mux(
                input_selector=get_value_at_index(seargegenerated1_340, 1),
                input0=get_value_at_index(emptylatentimage_44, 0),
                input1=get_value_at_index(vaeencode_282, 0),
                input2=get_value_at_index(setlatentnoisemask_283, 0),
            )

            seargesdxlsampler2_358 = seargesdxlsampler2.sample(
                noise_seed=fixed_seed,
                steps=get_value_at_index(seargeoutput2_359, 4),
                cfg=get_value_at_index(seargeoutput2_359, 5),
                base_ratio=get_value_at_index(seargeoutput3_369, 2),
                denoise=get_value_at_index(seargeoutput3_369, 1),
                refiner_prep_steps=get_value_at_index(seargeoutput3_369, 5),
                noise_offset=get_value_at_index(seargeoutput3_369, 4),
                refiner_strength=get_value_at_index(seargeoutput3_369, 3),
                base_model=get_value_at_index(seargeloraloader_353, 0),
                base_positive=get_value_at_index(seargeconditioningmuxer2_454, 0),
                base_negative=get_value_at_index(seargeconditioningmuxer2_455, 0),
                refiner_model=get_value_at_index(seargecheckpointloader_355, 0),
                refiner_positive=get_value_at_index(seargeconditioningmuxer2_456, 0),
                refiner_negative=get_value_at_index(seargeconditioningmuxer2_457, 0),
                latent_image=get_value_at_index(seargelatentmuxer3_335, 0),
                sampler_name=get_value_at_index(seargeoutput2_359, 6),
                scheduler=get_value_at_index(seargeoutput2_359, 7),
            )

            vaedecode_46 = vaedecode.decode(
                samples=get_value_at_index(seargesdxlsampler2_358, 0),
                vae=get_value_at_index(seargevaeloader_351, 0),
            )
            seargeimagesave.save_images(
                filename_prefix="transform_all/" + run + "/" + result_path,
                images=get_value_at_index(vaedecode_46, 0),
                state=get_value_at_index(seargeoutput2_359, 8),
                save_to=get_value_at_index(seargeoutput2_359, 9),
            )


if __name__ == "__main__":
    main()
