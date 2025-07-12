import os
import json
import base64
import requests
import torch

API_URL = "http://localhost:8000/v1/chat/completions"

PROMPT = """
Please describe the image of the plant leaf according to the following guidelines:
Step1,Identify the color of the leaf, including the base color and any phenotypic characteristics of spots or discolored areas, such as their location, size, length, number, and color.
Step2,describe the overall shape of the leaf, including whether it is round, oval, heart-shaped, or another shape.
Step3,describe the texture of the leaf, including whether the surface is smooth, hairy, or has other features.
Step4,if the leaf has edges, describe the characteristics of the edges, such as whether they are smooth, serrated, or wavy.
Step5,describe whether there are any visible damages on the leaf, such as holes, tears, wilting, or lesions.
Step6,if there are veins on the leaf, describe their distribution and color.
Step7,if the image includes the petiole of the leaf, describe the thickness, color, and texture of the petiole.
Please keep the description objective, providing only the information visible in the image without adding any inferences or explanations.
""".strip()


def encode_image(image_path):
    with open(image_path, "rb") as img_file:
        return f"data:image/jpeg;base64,{base64.b64encode(img_file.read()).decode('utf-8')}"


def describe_image(image_path):
    image_b64 = encode_image(image_path)
    payload = {
        "model": "cogvlm-chat-17b",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": image_b64}
                    },
                    {
                        "type": "text",
                        "text": PROMPT
                    }
                ]
            }
        ],
        "temperature": 0.0,
        "top_p": 1.0,
        "max_tokens": 1000,
        "stream": False
    }

    try:
        response = requests.post(API_URL, json=payload, timeout=60)
        result = response.json()
        return result["choices"][0]["message"]["content"] if "choices" in result else "Error"
    except Exception as e:
        return f"Exception: {str(e)}"


def batch_describe(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    for subfolder in os.listdir(input_dir):
        sub_path = os.path.join(input_dir, subfolder)
        if not os.path.isdir(sub_path):
            continue

        descriptions = {}
        print(f"正在处理目录: {subfolder}")

        for img_file in os.listdir(sub_path):
            if not img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                continue

            img_path = os.path.join(sub_path, img_file)
            print(f"处理图片: {img_file}")
            desc = describe_image(img_path)
            descriptions[img_file] = desc

            # 显存释放（模拟 decripe_train.py）
            torch.cuda.empty_cache()

        output_file = os.path.join(output_dir, f"{subfolder}.json")
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(descriptions, f, indent=2, ensure_ascii=False)

        print(f"{output_file}")


if __name__ == "__main__":
    # 输入图片根目录
    input_path = "./images/train/"
    # 输出 JSON 保存目录
    save_path = "./caption/train/"

    batch_describe(input_path, save_path)
