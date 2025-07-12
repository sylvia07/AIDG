import json
import os

# 定义label映射
label_mapping = {
    "Apple_leaf": 0,
    "Apple_rust_leaf": 1,
    "Apple_Scab_Leaf": 2,
    "Bell_pepper_leaf": 3,
    "Bell_pepper_leaf_spot": 4,
    "Blueberry_leaf": 5,
    "Cherry_leaf": 6,
    "Corn_Gray_leaf_spot": 7,
    "Corn_leaf_blight": 8,
    "Corn_rust_leaf": 9,
    "grape_leaf": 10,
    "grape_leaf_black_rot": 11,
    "Peach_leaf": 12,
    "Potato_leaf_early_blight": 13,
    "Potato_leaf_late_blight": 14,
    "Raspberry_leaf": 15,
    "Soyabean_leaf": 16,
    "Squash_Powdery_mildew_leaf": 17,
    "Strawberry_leaf": 18,
    "Tomato_Early_blight_leaf": 19,
    "Tomato_leaf": 20,
    "Tomato_leaf_bacterial_spot": 21,
    "Tomato_leaf_late_blight": 22,
    "Tomato_leaf_mosaic_virus": 23,
    "Tomato_leaf_yellow_virus": 24,
    "Tomato_mold_leaf": 25,
    "Tomato_Septoria_leaf_spot": 26,
    "Tomato_two_spotted_spider_mites_leaf": 27
}
# label_mapping = {
#     "Apple_leaf": 0,
#     "Apple_rust_leaf": 1,
#     "Apple_Scab_Leaf": 2,
# }
# 读取data目录下的所有JSON文件并添加label
def add_label_to_json_files(data_directory):
    # 获取data目录下所有JSON文件
    json_files = [f for f in os.listdir(data_directory) if f.endswith('.json')]

    # 遍历所有JSON文件进行处理
    for json_file in json_files:
        input_file_path = os.path.join(data_directory, json_file)
        output_file_path = os.path.join(data_directory+"_label", f"{json_file}")
        print(output_file_path)
        # 获取文件夹名称作为label值
        folder_name = json_file.split('.')[0]  # 使用文件名作为文件夹标识
        label_value = label_mapping.get(folder_name, None)
        
        # 检查label值是否存在
        if label_value is None:
            print(f"Warning: Label value for '{folder_name}' is not defined in the label mapping.")
            continue  # 跳过没有定义label值的文件
        
        # 读取原始JSON文件
        with open(input_file_path, 'r') as file:
            data = json.load(file)

        # 修改JSON格式，添加label标签
        modified_data = {}
        for image, description in data.items():
            modified_data[image] = {
                "text": description,
                "label": label_value
            }

        # 将修改后的数据写入新的JSON文件
        with open(output_file_path, 'w') as file:
            json.dump(modified_data, file, indent=4)
        
        print(f"Modified JSON for {json_file} has been written to {output_file_path}")

# 设定data目录路径
data_directory = "/data/LLAVA_main/llava/eval/different_prompt/cogagent/captions/train"

# 执行处理
add_label_to_json_files(data_directory)
