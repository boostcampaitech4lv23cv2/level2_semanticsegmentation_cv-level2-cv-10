import os
import json
import shutil

DATAROOT = "/opt/ml/input/data/"
TRAINJSON = os.path.join(DATAROOT, "train.json")
VALIDJSON = os.path.join(DATAROOT, "val.json")
TESTJSON = os.path.join(DATAROOT, "test.json")


def chg_dir(json, path):
    imagePath = "/opt/ml/input/data/" + path
    os.makedirs(imagePath, exist_ok=True)
    
    with open(json, "r", encoding="utf8") as outfile:
        json_data = json.load(outfile)
    image_datas = json_data["images"]

    for image_data in image_datas:
        shutil.copyfile(
            os.path.join("/opt/ml/input/data", image_data["file_name"]),
            os.path.join(imagePath, f"{image_data['id']:04}.jpg"),
        )


json_list = [TRAINJSON, VALIDJSON, TESTJSON]
path_list = ['images/train', 'images/val', 'test']
             
for json, path in zip(json_list, path_list):
    chg_dir(json, path)