
# import scipy.io as sio
# import numpy as np
from collections import Counter

# # 读取.mat文件
# mat_data = sio.loadmat('./datasets/ORIGA/test/mask/AGLAIA_GT_001.mat')

# # 提取变量
# # matrix1 = mat_data['matrix1']
# # matrix2 = mat_data['matrix2']

# # 显示变量信息
# # i = 1
# # for key, value in mat_data.items():
# #     print(i)
# #     print("key:",key)
# #     print("value:",value)
# #     i+=1
# #     # print(value)

# origin_mask = mat_data['maskFull']
# print('Counter(data)\n',Counter(origin_mask.flatten()))
# print(mat_data['maskFull'])
# print(type(origin_mask))
# import torch
# from seg_train import *
# from seg_train import ORIGAbase
# import segmentation_models_pytorch as smp
# from safetensors.torch import load_file
# import matplotlib.pyplot as plt
import PIL
from PIL import Image
# model = smp.Unet(
#     encoder_name='resnet18',           # choose encoder, e.g. mobilenet_v2 or efficientnet-b0
#     encoder_weights="imagenet",     # use `imagenet` pretrained weights for encoder initialization
#     in_channels=3,                  # model input channels (1 for grayscale images, 3 for RGB, etc.)
#     classes=3,                      # model output channels (number of classes in your dataset)
# )


# # transform = transforms.Compose([
# #     transforms.RandomHorizontalFlip(p=0.5),
# #     transforms.RandomVerticalFlip(p=0.5),
# #     # transforms.CenterCrop(size=(256, 256))
# # ])


# model.load_state_dict(load_file('./model-finetuned/checkpoint-80-0.8643504977226257--0.9921613729000092/model.safetensors'))
 
# # mask_path =  "./demo/mask.jpg"  
# image = Image.open(image_path).convert("RGB")
# image = image.resize((512, 512), resample=PIL.Image.BILINEAR)
import numpy as np
image_path = "./datasets/G1020/test/mask/image_0.png"  
# 读取 PNG 图像（灰度模式）


image = Image.open(image_path).convert("L")

# 转换为 NumPy 数组
array = np.array(image)
print('Counter(data)\n',Counter(array.flatten()))
# 创建映射关系：0 -> 0, 1 -> 128, 2 -> 255
mapping = {0: 0, 1: 128, 2: 255}
vectorized_map = np.vectorize(lambda x: mapping.get(x, x))

# 应用映射
new_array = vectorized_map(array).astype(np.uint8)

# 转换回 PIL 图像
new_image = Image.fromarray(new_array)

# 保存新的图像
new_image.save("converted_image.png")

# 显示图像（可选）
new_image.show()

# image = np.array(image).astype(np.float16) / 255.0
# image = torch.Tensor(image).permute(2, 0, 1) 
# image = image.to(torch.float32)
# # mask = sample["mask"].to(weight_dtype)
# image = image.unsqueeze(0)
# reconstructions = model(image)
# # mask = F.one_hot(mask.long(), num_classes).permute(0, 3, 1, 2).float()
# reconstructions = F.one_hot(reconstructions.argmax(dim=1), 3).permute(0, 3, 1, 2).float()
# plt.imsave('./demo/predict.png', np.transpose(np.array(reconstructions[0]), (1, 2, 0)))




# ###############################################
# import cv2
# import numpy as np
# from PIL import Image
# # 读取图像
# # image = Image.open('.datasets/g0002.jpg')
# # image = cv2.imread('./datasets/g0002.jpg', cv2.IMREAD_GRAYSCALE)
# def Ta(image):
#     # 1. 转换 PIL 图像为 NumPy 数组（灰度图像）
#     image = np.array(image.convert('L'))  # 'L' 模式转换为灰度图像

#     # 确保图像是 8 位单通道（uint8）
#     image = np.uint8(image)

#     # 2. 剔除背景（去除黑色像素）
#     mask = image > 0  # 创建一个掩码，将非零部分保留下来

#     # 使用掩码去除背景
#     image_no_background = np.zeros_like(image)
#     image_no_background[mask] = image[mask]

#     # 3. 提取眼底区域：假设眼底部分是圆形，可以通过轮廓检测提取该区域
#     contours, _ = cv2.findContours(image_no_background, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

#     # 假设最大的轮廓是眼底区域
#     max_contour = max(contours, key=cv2.contourArea)

#     # 计算圆形区域的最小外接矩形
#     x, y, w, h = cv2.boundingRect(max_contour)

#     # 剪切出眼底区域
#     eye_region = image_no_background[y:y+h, x:x+w]

#     # 4. 将圆形区域裁剪为正方形
#     side_length = max(w, h)  # 选择裁剪后的区域宽度和高度为较大的边

#     # 创建一个正方形画布（背景填充为0）
#     square_image = np.zeros((side_length, side_length), dtype=np.uint8)

#     # 计算裁剪区域的起始位置
#     start_x = (side_length - w) // 2
#     start_y = (side_length - h) // 2

#     # 将圆形区域填充到正方形画布中
#     square_image[start_y:start_y+h, start_x:start_x+w] = eye_region

#     # 5. 将正方形区域 resize 为所需的尺寸
#     final_image = cv2.resize(square_image, (512, 512))  # 例如 resize 到 512x512

#     # 将处理后的 NumPy 数组转换回 PIL Image
#     final_image_pil = Image.fromarray(final_image)

#     # 返回最终的 PIL 图像
#     return final_image_pil
