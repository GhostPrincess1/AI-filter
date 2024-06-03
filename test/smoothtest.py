import numpy as np

def adaptive_filter(frame1_coords, frame2_coords, threshold, alpha):
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

# 假设这两个列表包含frame1和frame2中的坐标点
frame1_coordinates = [np.array([1, 2]), np.array([3, 4])]
frame2_coordinates = [np.array([3, 2]), np.array([20, 20])]

# 设置阈值和加权平均法的权重
threshold = 5
alpha = 0.1

filtered_coordinates = adaptive_filter(frame1_coordinates, frame2_coordinates, threshold, alpha)
print(filtered_coordinates)
