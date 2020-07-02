from xml.dom.minidom import parse
import math, os

# lable = r"D:\BaiduNetdiskDownload\allCode\yolov3_teach\yolov3_teach\data\person_label.txt"
# lable_path = r"D:\BaiduNetdiskDownload\allCode\yolov3_teach\yolov3_teach\data\images\outputs"
lable = r"D:\pycharmworkspace\DeepLearningStudy\MediateCourse\yolo_V3\data\animal_label.txt"
lable_path = r"D:\Alldata\outputs"

file_names = os.listdir(lable_path)
labelTxt = open(lable, "w")
for filename in file_names:
    dom = parse(f"{lable_path}/{filename}")
    root = dom.documentElement
    img_name = root.getElementsByTagName("path")[0].childNodes[0].data
    objects = root.getElementsByTagName("object")
    for box in objects:
        strs = img_name.split("\\")
        path = strs[-2] + "/" + strs[-1]
        labelTxt.write(f"{path}")
        items = box.getElementsByTagName("item")
        for item in items:
            cls_name = item.getElementsByTagName("name")[0].childNodes[0].data
            x1 = int(item.getElementsByTagName("xmin")[0].childNodes[0].data)
            y1 = int(item.getElementsByTagName("ymin")[0].childNodes[0].data)
            x2 = int(item.getElementsByTagName("xmax")[0].childNodes[0].data)
            y2 = int(item.getElementsByTagName("ymax")[0].childNodes[0].data)
            w, h = x2 - x1, y2 - y1
            cx, cy = math.ceil(x1 + w / 2), math.ceil(y1 + h / 2)

            if cls_name == "猫":
                cls_name = 0
            elif cls_name == "狗":
                cls_name = 1
            elif cls_name == "人":
                cls_name = 2
            labelTxt.write(f" {cls_name} {cx} {cy} {w} {h}")

        labelTxt.write("\n")
        labelTxt.flush()
