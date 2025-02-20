import imageio
import os

# 定义图像文件的路径和文件名列表
image_folder = '/home/huang/code/ROS_SDK/proj_ws/src/TCP-IP-ROS-6AXis-main/dobot_bringup/script/results/real4_open/images_1'  # 替换为你的图像文件夹路径
image_files = sorted([f for f in os.listdir(image_folder) if f.endswith(('png', 'jpg', 'jpeg'))], key=lambda s: int(s[:-4]))

# 读取图像文件并存储在列表中
images = [imageio.imread(os.path.join(image_folder, file)) for file in image_files]

# 定义GIF文件的保存路径和文件名
gif_path = '/home/huang/code/ROS_SDK/proj_ws/src/TCP-IP-ROS-6AXis-main/dobot_bringup/script/results/real4_open/images_1.gif'  # 替换为你想要保存的GIF文件名

# 将图像列表保存为GIF文件
imageio.mimsave(gif_path, images, duration=0.5)  # duration参数设置每张图像显示的持续时间（秒）

print(f"GIF文件已保存到 {gif_path}")
