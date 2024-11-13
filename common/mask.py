import torch
from PIL import Image, ImageDraw, ImageFilter
import os
import re

def load_coordinates(file_path):
    """加载坐标文件"""
    data = torch.load(file_path)  # 加载 PyTorch .pth 文件
    return data

def apply_mask(image_path, coordinates, output_path):
    """遮盖指定区域并保存结果图像"""
    # 加载图像
    image = Image.open(image_path).convert('RGB')
    draw = ImageDraw.Draw(image)
    
    # 图像的宽度和高度
    width, height = image.size

    # 遍历坐标，遮盖区域
    x_center, y_center, w, h, s = coordinates[0], coordinates[1], coordinates[2], coordinates[3], coordinates[4]

    # 计算左上角和右下角的坐标
    x_min = (x_center - w / 2) * width
    y_min = (y_center - h / 2) * height
    x_max = (x_center + w / 2) * width
    y_max = (y_center + h / 2) * height

    # 确保坐标在图像尺寸范围内
    x_min = max(0, min(width, x_min))
    y_min = max(0, min(height, y_min))
    x_max = max(0, min(width, x_max))
    y_max = max(0, min(height, y_max))

    # 绘制遮盖区域
    draw.rectangle([x_min, y_min, x_max, y_max], fill="black")  # 使用黑色遮盖

    # 保存结果图像
    #image.save(output_path)


# def apply_concentrate(image_path, coordinates, output_path):
#     """遮盖指定区域并保存结果图像"""
#     # 加载图像
#     image = Image.open(image_path).convert('RGB')
#     width, height = image.size
    
#     # 创建一个全黑的遮盖图像
#     if flag == 0:
#         mask_image = Image.new('RGBA', (width, height), color=(0, 0, 0, 255))
#         mask_draw = ImageDraw.Draw(mask_image)
#     else:
#         mask_draw =  ImageDraw.Draw(image)
    
#     # 遍历坐标，绘制遮盖区域
#     x_center, y_center, w, h, s = coordinates[0], coordinates[1], coordinates[2], coordinates[3], coordinates[4]

#     # 计算左上角和右下角的坐标
#     x_min = int((x_center - w / 2) * width)
#     y_min = int((y_center - h / 2) * height)
#     x_max = int((x_center + w / 2) * width)
#     y_max = int((y_center + h / 2) * height)

#     # 确保坐标在图像尺寸范围内
#     x_min = max(0, min(width, x_min))
#     y_min = max(0, min(height, y_min))
#     x_max = max(0, min(width, x_max))
#     y_max = max(0, min(height, y_max))

#     # 绘制遮盖区域
#     mask_draw.rectangle([x_min, y_min, x_max, y_max], fill=(255, 255, 255, 0))  # 透明白色遮盖

#     # 创建一个新的图像，将遮盖应用到原始图像上
#     masked_image = Image.new('RGBA', (width, height), color=(0, 0, 0, 0))
#     masked_image.paste(image, (0, 0))
#     masked_image.paste(mask_image, (0, 0), mask_image)

#     # 转换为 RGB 模式并保存结果图像
#     final_image = masked_image.convert('RGB')
#     final_image.save(output_path)
    
def process_images(coordinates_file_path, image_dir, output_dir):
    """处理所有图像，应用遮盖并保存"""
    coordinates = load_coordinates(coordinates_file_path)
    print(f"加载的坐标数量: {len(coordinates)}")
    d = []
    # 遍历坐标数据
    for image_filename, boxes in coordinates.items():
        # 去掉 _crop_1 后缀
        image_filename = re.sub(r'_crop.*\.jpg$', '.jpg', image_filename)
        # 构造图像路径
        image_path = os.path.join(image_dir, image_filename)
        output_path = os.path.join(output_dir, image_filename)
        if image_filename not in d:
            # 应用遮盖
            apply_mask(image_path, boxes, output_path)
        # 应用遮盖
        else:
            apply_mask(output_path, boxes, output_path)
        d.append(image_filename)



def apply_concentrate(image_path, coordinates, output_path):
    """遮盖指定区域并保存结果图像"""
    # 加载图像
    image = Image.open(image_path).convert('RGB')
    width, height = image.size

    # 创建一个全透明的遮盖图像
    mask_image = Image.new('RGBA', (width, height), color=(0, 0, 0, 0))
    mask_draw = ImageDraw.Draw(mask_image)

    # 创建一个全白的背景图像，用来标记不需要模糊的区域
    keep_image = Image.new('RGBA', (width, height), color=(255, 255, 255, 255))

    # 绘制需要保留的区域（即给定的坐标区域）
    for box in coordinates:
        x_center, y_center, w, h, s = box[0], box[1], box[2], box[3], box[4]
        # 计算左上角和右下角的坐标
        x_min = int((x_center - w / 2) * width)
        y_min = int((y_center - h / 2) * height)
        x_max = int((x_center + w / 2) * width)
        y_max = int((y_center + h / 2) * height)

        # 确保坐标在图像尺寸范围内
        x_min = max(0, min(width, x_min))
        y_min = max(0, min(height, y_min))
        x_max = max(0, min(width, x_max))
        y_max = max(0, min(height, y_max))

        # 绘制保留区域到遮盖图像
        mask_draw.rectangle([x_min, y_min, x_max, y_max], fill=(0, 0, 0, 255))  # 不透明

    # 将原图模糊处理
    blurred_image = image.filter(ImageFilter.GaussianBlur(radius=10))

    # 将模糊处理图像粘贴到结果图像
    result_image = Image.new('RGBA', (width, height), color=(0, 0, 0, 0))
    result_image.paste(blurred_image, (0, 0))

    # 将需要保留的区域从原图拷贝到结果图像
    result_image.paste(image.convert('RGBA'), (0, 0), mask_image)

    # 转换为 RGB 模式并保存结果图像
    final_image = result_image.convert('RGB')
    final_image.save(output_path)
    # 反转遮盖图像：将需要遮盖的区域从全黑的遮盖图像中排除
    # mask_image = Image.new('RGBA', (width, height), color=(0, 0, 0, 255))
    # mask_image.paste(Image.new('RGBA', (width, height), color=(0, 0, 0, 0)), (0, 0), mask_image)

    # # 将遮盖图像应用于原图像
    # final_image = Image.alpha_composite(image.convert('RGBA'), mask_image)

    # # 转换为 RGB 模式并保存结果图像
    # final_image.convert('RGB').save(output_path)

def collect_boxes(coordinates):
    """收集每个图像的所有 boxes"""
    image_boxes = {}
    for image_filename, boxes in coordinates.items():
        print(boxes)
        # 去掉 _crop 后缀
        image_filename = re.sub(r'_crop.*\.jpg$', '.jpg', image_filename)
        if image_filename not in image_boxes:
            image_boxes[image_filename] = []
        image_boxes[image_filename].append(boxes)
    return image_boxes

def process_images_concentrate(coordinates_file_path, image_dir, output_dir):
    """处理所有图像，应用遮盖并保存"""
    coordinates = load_coordinates(coordinates_file_path)
    print(f"加载的坐标数量: {len(coordinates)}")
    
    # 收集每个图像的所有 boxes
    image_boxes = collect_boxes(coordinates)

    # 遍历每个图像并应用遮盖
    for image_filename, boxes in image_boxes.items():
        # 构造图像路径和输出路径
        image_path = os.path.join(image_dir, image_filename)
        output_path = os.path.join(output_dir, image_filename)

        # 应用遮盖
        apply_concentrate(image_path, boxes, output_path)
        
        print(f"遮盖后的图片已保存到: {output_path}")
    

# 示例使用
coordinates_file_path = '/home/nfs02/xingsy/code/data/MNRE/mnre_image/region/test_detect_box_dict.pth'
image_dir = '/home/nfs02/xingsy/code/data/MNRE/mnre_image/img_org/test'  # 替换为实际图像目录路径
output_dir = '/home/nfs02/xingsy/code/data/MNRE/mnre_image/img_org/new_concentrate'  # 替换为输出图像目录路径

# 处理所有图像
# process_images(coordinates_file_path, image_dir, output_dir)

process_images_concentrate(coordinates_file_path, image_dir, output_dir)