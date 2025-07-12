import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append('/data/LLAVA_main/')

from llava.model.builder import load_pretrained_model
from mm_utils import get_model_name_from_path
from run_llava import eval_model
import json

model_path = "/data/LLAVA_main/MODEL/llava-v1.6-vicuna-7b"

def simple_image_chat(img_path):
    prompt = """
Please describe the image of the plant leaf according to the following guidelines:
Step1,Identify the color of the leaf, including the base color and any phenotypic characteristics of spots or discolored areas, such as their location, size, length, number, and color.
Step2,describe the overall shape of the leaf, including whether it is round, oval, heart-shaped, or another shape.
Step3,describe the texture of the leaf, including whether the surface is smooth, hairy, or has other features.
Step4,if the leaf has edges, describe the characteristics of the edges, such as whether they are smooth, serrated, or wavy.
Step5,describe whether there are any visible damages on the leaf, such as holes, tears, wilting, or lesions.
Step6,if there are veins on the leaf, describe their distribution and color.
Step7,if the image includes the petiole of the leaf, describe the thickness, color, and texture of the petiole.
Please keep the description objective, providing only the information visible in the image without adding any inferences or explanations.
"""
    image_file = img_path

    args = type('Args', (), {
        "model_path": model_path,
        "model_base": None,
        "model_name": get_model_name_from_path(model_path),
        "query": prompt,
        "conv_mode": None,
        "image_file": image_file,
        "sep": ",",
        "temperature": 0,
        "top_p": None,
        "num_beams": 1,
        "max_new_tokens": 1000
    })()
    print(image_file)
    return eval_model(args)

def generate_descriptions(path):
    descriptions = {}
    for image_name in os.listdir(path):
        image_path = path + "/" + image_name
        descriptions[image_name] = simple_image_chat(img_path=image_path)
        print('*'*100)
        print(descriptions[image_name])
        print('*'*100)
    return descriptions

def get_twitter_descriptions(image_path):
    for fold in os.listdir(image_path):
        result = {}
        file_path = image_path + fold
        descriptions = generate_descriptions(file_path)
        result.update(descriptions)
        save_path = "./different_prompt/tomato_caption/" + fold + ".json"
        with open(save_path, "w+") as f:
             json.dump(result, f)
image_path = "./different_prompt/tomato_train/"
# image_path = "./image/test/"
get_twitter_descriptions(image_path)



