import os
import json
import glob
import xml.etree.ElementTree as ET
from tqdm import tqdm

START_BOUNDING_BOX_ID = 1
PRE_DEFINE_CATEGORIES = {
    "Korra": 1,
    "Mako": 2
}

def get(root, name):
    return root.find(name).text

def convert(xml_list, image_dir, output_json_path):
    json_dict = {
        "info": {
            "description": "Avatar dataset_test",
            "version": "1.0",
            "year": 2025
        },
        "images": [],
        "type": "instances",
        "annotations": [],
        "categories": []
    }

    for k, v in PRE_DEFINE_CATEGORIES.items():
        cat = {
            "supercategory": "none",
            "id": v,
            "name": k
        }
        json_dict["categories"].append(cat)

    bnd_id = START_BOUNDING_BOX_ID
    image_id = 1  # 新增：全局唯一图片ID计数器

    for xml_file in tqdm(xml_list):
        tree = ET.parse(xml_file)
        root = tree.getroot()
        filename = get(root, "filename")
        # 取消原先从文件名提取id的方法，改为使用image_id计数器
        # image_id = int(os.path.splitext(filename)[0].split('_')[-1])

        size = root.find("size")
        width = int(get(size, "width"))
        height = int(get(size, "height"))

        image = {
            "file_name": filename,
            "height": height,
            "width": width,
            "id": image_id  # 使用计数器id
        }
        json_dict["images"].append(image)

        for obj in root.findall("object"):
            category = get(obj, "name")
            if category not in PRE_DEFINE_CATEGORIES:
                continue

            cat_id = PRE_DEFINE_CATEGORIES[category]
            bndbox = obj.find("bndbox")
            xmin = int(get(bndbox, "xmin"))
            ymin = int(get(bndbox, "ymin"))
            xmax = int(get(bndbox, "xmax"))
            ymax = int(get(bndbox, "ymax"))
            o_width = max(0, xmax - xmin)
            o_height = max(0, ymax - ymin)

            ann = {
                "area": o_width * o_height,
                "iscrowd": 0,
                "image_id": image_id,  # 关联的图片id
                "bbox": [xmin, ymin, o_width, o_height],
                "category_id": cat_id,
                "id": bnd_id,
                "ignore": 0,
                "segmentation": []
            }
            json_dict["annotations"].append(ann)
            bnd_id += 1

        image_id += 1  # 图片id计数器自增

    with open(output_json_path, 'w') as json_fp:
        json.dump(json_dict, json_fp, indent=4)

if __name__ == "__main__":
    xml_dir = r"D:\Desktop\AI\project\annotations\xmls_val"
    image_dir = r"D:\Desktop\AI\project\images\val"
    output_path = r"D:\Desktop\AI\project\annotations\instances_val.json"

    xml_files = glob.glob(os.path.join(xml_dir, "*.xml"))
    convert(xml_files, image_dir, output_path)
