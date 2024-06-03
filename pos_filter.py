import copy
from scipy.interpolate import splprep, splev
import numpy as np
def get_index_points(pos_points,index):
    points_index  = []
    for i in range(len(pos_points)):
        points_index.append(pos_points[i][index])
    return points_index


def get_index_points_xy(points_index):
    points_index_x,points_index_y = [],[]

    for i in range(len(points_index)):
        points_index_x.append(points_index[i][0])
        points_index_y.append(points_index[i][1])
    
    return points_index_x,points_index_y

def myfilter(data):
    x = np.arange(len(data))

    # 计算B样条曲线
    tck, u = splprep([x, data], s=100,k=2)
    new_x, new_data = splev(np.linspace(0, 1, len(data)), tck)
    for i in range(len(new_data)):
        new_data[i] = int(new_data[i])
    return new_data


# input pos_points
def get_filtered_points(pos_points_frames):


    pos_points = copy.deepcopy(pos_points_frames)
    #获取某索引值的二维列表

    index = len(pos_points[0])-1
    print(index)

    for i in range(index+1):

        points_index = get_index_points(pos_points,i)

        #获取x轴坐标

        points_index_x,points_index_y = get_index_points_xy(points_index)
      

        new_x = myfilter(points_index_x)
        

        new_y = myfilter(points_index_y)
      

        for j in range(len(points_index)):
            points_index[j][0] = new_x[j]
            points_index[j][1] = new_y[j]
        
        for j in range(len(pos_points)):
            pos_points[j][i] = points_index[j]

    return pos_points


    
if __name__ == "__main__":

    pos_points_frames = [[np.array([2,9]),np.array([9,125]),np.array([1,5])],
                         [np.array([80,15]),np.array([100,45]),np.array([5,9])],
                         [np.array([90,14]),np.array([19,65]),np.array([54,42])],
                         [np.array([180,150]),np.array([250,900]),np.array([2,8])]]

    re = get_filtered_points(pos_points_frames)
    print(re)
    pass