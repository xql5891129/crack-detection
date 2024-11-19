import os
import cv2
import json
import numpy as np

def json_to_png(json_file, output_folder):
    with open(json_file, 'r') as f:
        data = json.load(f)
    # 获取图像尺寸
    img_width = data['imageWidth']
    img_height = data['imageHeight']
    # 创建空白图像
    image = np.zeros((img_height, img_width), dtype=np.uint8)
    # 遍历每个标注对象
    for shape in data['shapes']:
        label = shape['label']
        points = shape['points']
        # 将多边形点转换为NumPy数组
        points = np.array(points, dtype=np.int32)
        # 在图像上绘制多边形
        cv2.fillPoly(image, [points], color=255)
    # 构造输出PNG文件的路径
    output_file = os.path.join(output_folder, os.path.splitext(os.path.basename(json_file))[0] + '.png')
    # 保存图像为PNG格式
    cv2.imwrite(output_file, image)
# 输入JSON文件夹路径和输出PNG文件夹路径
json_folder = 'json'
output_folder = 'png'
# 遍历JSON文件夹中的每个文件
for file_name in os.listdir(json_folder):
    if file_name.endswith('.json'):
        # 构造JSON文件的路径
        json_file = os.path.join(json_folder, file_name)
        # 将JSON转换为PNG并保存到输出文件夹
        json_to_png(json_file, output_folder)
