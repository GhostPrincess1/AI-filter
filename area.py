import numpy as np

def calculate_quadrilateral_area(points):
    if len(points) != 4:
        raise ValueError("Input should contain exactly 4 points.")
    
    x1, y1 = points[0]
    x2, y2 = points[1]
    x3, y3 = points[2]
    x4, y4 = points[3]

    area = 0.5 * abs(x1*(y2-y3) + x2*(y3-y1) + x3*(y1-y2) + x2*(y3-y4) + x4*(y2-y3))
    return area

def calculate_distance(point1, point2):
    distance = np.linalg.norm(point2 - point1)
    return distance



if __name__ == "__main__":

    # 输入四个点的坐标，每个点都是一个NumPy数组
    points = np.array([
        [1.0, 2.0],
        [3.0, 4.0],
        [5.0, 6.0],
        [7.0, 8.0]
    ])

    # 调用函数计算面积
    area = calculate_quadrilateral_area(points)

    # 打印结果
    print("四边形的面积为:", area)

