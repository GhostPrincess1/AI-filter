import numpy as np
import cv2
import os
import copy

    
def find_point_C(A, B, b):
    """
    计算满足AB与AC共线且AC长度为b的点C的坐标。

    参数：
    A: 点A的坐标，例如 (x1, y1)
    B: 点B的坐标，例如 (x2, y2)
    b: 固定长度AC的长度

    返回：
    点C的坐标，例如 (x3, y3)
    """
    # 计算AB向量
    AB_vector = B - A
    
    # 计算AB向量的长度
    AB_length = np.linalg.norm(AB_vector)
    
    # 计算单位向量u
    if AB_length == 0:
        raise ValueError("点A和点B在同一位置，无法计算单位向量。")
    u = AB_vector / AB_length
    
    # 计算点C的坐标
    C = A + b * u
    
    return C

def gmp4():
    # 指定输入文件夹和输出文件名
    input_folder = 'images'
    output_video = 'output_video.mp4'

    # 获取输入文件夹中所有的PNG图像文件
    image_files = [f for f in os.listdir(input_folder) if f.endswith('.png')]
    image_files.sort()  # 确保按顺序合并帧

    # 获取第一帧图像的尺寸（假设所有图像具有相同的尺寸）
    frame = cv2.imread(os.path.join(input_folder, image_files[0]))
    height, width, layers = frame.shape

    # 设置视频编码器和输出视频
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 使用MP4编码
    video = cv2.VideoWriter(output_video, fourcc, 30, (width, height))  

    # 合并PNG序列帧到视频
    for image_file in image_files:
        image_path = os.path.join(input_folder, image_file)
        frame = cv2.imread(image_path)
        video.write(frame)

    # 完成后释放资源
    video.release()

    print(f'视频已保存为: {output_video}')

    return output_video
def get_ith_file_in_folder(folder_path, i):
    # 列出文件夹中的所有文件
    files = os.listdir(folder_path)
    print(len(files))

    # 过滤出文件，排除文件夹和子文件夹
    files = [f for f in files if os.path.isfile(os.path.join(folder_path, f))]

    # 检查是否存在第i个文件
    if i < len(files):
        ith_file = os.path.join(folder_path, files[i])
        return ith_file
    else:
        return None


def extract_frames_from_video(video_path, output_folder):
    # 打开视频文件
    cap = cv2.VideoCapture(video_path)

    # 创建保存序列帧的目录
    os.makedirs(output_folder, exist_ok=True)

    # 读取并保存序列帧
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 生成序列帧文件名
        frame_filename = os.path.join(output_folder, f'frame_{frame_count:04d}.png')

        # 保存帧为图像文件
        cv2.imwrite(frame_filename, frame)

        frame_count += 1

    # 关闭视频文件
    cap.release()

    print(f"成功保存了 {frame_count} 个序列帧到 {output_folder} 目录。")


def create_green_blg(height,width):
    # 创建一个全绿色的帧
    green_color = (0, 255, 0)  # 绿色 (BGR 格式)
    green_frame = np.zeros((height, width, 3), dtype=np.uint8)
    green_frame[:] = green_color

    return green_frame

def get_norm_vector(vector):
        # 计算向量的模
    magnitude = np.linalg.norm(vector)
        # 计算单位向量
    unit_vector = vector / magnitude

    return unit_vector

def get_destlength_vector(d,norm_vector):

    return d*norm_vector
    

def rule_points(reference_vector,points):

    #tartget_root_point是y轴不变点
    temp_points = copy.deepcopy(points)

    for i in range(len(temp_points)):
        temp_points[i] = temp_points[i] + reference_vector


    return temp_points
    
def get_target_point_frome_blg(frame):

    height,width,layer = frame.shape
    x = (0.1)*width
    x = int(x)

    y = (0.9)*height
    y = int(y)

    return np.array([x,y])

def adaptive_filter(frame1_coords, frame2_coords, threshold = 2, alpha = 0.1):
    filtered_coords = []
    for coord1, coord2 in zip(frame1_coords, frame2_coords):
        distance = np.linalg.norm(coord1 - coord2)
        if distance < threshold:
            filtered_coord = np.add(np.multiply(coord1, 1 - alpha), np.multiply(coord2, alpha))
        else:
            filtered_coord = coord2
        filtered_coords.append(filtered_coord)
    for i in range(len(filtered_coords)):
        filtered_coords[i] = filtered_coords[i].astype("int")
        

    return filtered_coords

if __name__ == "__main__":
        # 假设这两个列表包含frame1和frame2中的坐标点
    frame1_coordinates = [np.array([1, 2]), np.array([3, 4])]
    frame2_coordinates = [np.array([3, 2]), np.array([20, 20])]

    # 设置阈值和加权平均法的权重
    threshold = 5
    alpha = 0.1

    filtered_coordinates = adaptive_filter(frame1_coordinates, frame2_coordinates, threshold, alpha)
    print(filtered_coordinates)