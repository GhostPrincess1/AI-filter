import cv2
import numpy as np

def perspectiveTransform(frame1, frame2, points1, points2):
    

    # 读取两个帧的图像
    image1 = cv2.imread(frame1)
    image2 = cv2.imread(frame2)

    # 定义源图像上的三个坐标点
    src_points = np.float32(points2)

    # 定义目标图像上的对应的三个坐标点
    dst_points = np.float32(points1)

    # 如果只提供了三个坐标点，虚构第四个点
    if len(src_points) == 3:
        # 计算两个已知对应点之间的中点
        x_mid = (src_points[0][0] + src_points[1][0]) / 2
        y_mid = (src_points[0][1] + src_points[1][1]) / 2

        # 计算第三个已知点到中点的向量
        x_vec = src_points[2][0] - x_mid
        y_vec = src_points[2][1] - y_mid

        # 使用中点和向量虚构第四个点
        src_points = np.vstack((src_points, [x_mid + x_vec, y_mid + y_vec]))

    # 计算透视变换矩阵
    M = cv2.getPerspectiveTransform(src_points, dst_points)
    print(M)

    # 应用透视变换将第二个帧映射到第一个帧的坐标点上
    result = cv2.warpPerspective(image2, M, (image1.shape[1], image1.shape[0]))

    # 叠加两个帧的图像
    final_result = cv2.addWeighted(image1, 1, result, 1, 0)

    return final_result


# 示例用法
frame1 = r'F:\Create-AR-filters-using-Mediapipe\ench_body\MainWeapon_s2.png'
frame2 = r'F:\Create-AR-filters-using-Mediapipe\ench_body\static_head.png'
points1 = [[91, 394], [425, 479], [910, 601],[500,480]]
points2 = [[10, 50], [300, 150], [150, 10],[80,167]]

result = perspectiveTransform(frame1, frame2, points1, points2)

# 显示结果
cv2.imshow('Result', result)
cv2.waitKey(0)
cv2.destroyAllWindows()
