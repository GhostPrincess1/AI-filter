import cv2


def resize_picture(image_path):

    
    # 读取PNG图像
    image = cv2.imread(image_path)

    # 获取图像的高度和宽度
    height, width = image.shape[:2]

    # 计算缩放因子
    vertical_scale = 1.2  # 垂直方向放大30%
    horizontal_scale = 1.1  # 水平方向放大20%

    # 使用cv2.resize进行缩放
    resized_image = cv2.resize(image, (int(width * horizontal_scale), int(height * vertical_scale)))

    # 保存结果
    cv2.imwrite(image_path, resized_image)


if __name__ == "__main__":

    resize_picture("ench\Eyes.png")
    
