import cv2
import numpy as np


img1 = cv2.imread("ench_body\MainWeapon_s2.png")
frame = cv2.imread("test\\7342.jpg_wh300.jpg")
# 定义两组对应点
src_points = np.array([[0, 0], [100, 0], [0, 100]])
dst_points = np.array([[91,394], [425, 479], [910, 601]])

# 估计部分仿射变换矩阵
affine_matrix = cv2.estimateAffinePartial2D(src_points, dst_points)

# 用估计的变换矩阵进行变换
trans_img = cv2.warpAffine(img1, affine_matrix[0], (frame.shape[1], frame.shape[0]))

result = trans_img+frame

# 显示结果
cv2.imshow('Result', trans_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
