import mediapipe as mp
import cv2
import math
import numpy as np
import faceBlendCommon as fbc
import csv
from area import calculate_quadrilateral_area
from area import calculate_distance
from pos_filter import get_filtered_points
from utils import find_point_C, get_target_point_frome_blg
from utils import create_green_blg,adaptive_filter
from pos_retartget import PosRetarget
import copy
from flask import render_template
from flask import Flask, request, send_file
import os
from threading import Semaphore
import datetime
from utils import rule_points
import sys
from black_ground import Blg



app = Flask(__name__)
VISUALIZE_FACE_POINTS = False

#rule_points = [np.array([819, 644]), np.array([851, 545]), np.array([888, 526]), np.array([918, 555]), np.array([770, 539]), np.array([725, 520]),np.array([676, 545]), np.array([934, 587]), np.array([644, 575]), np.array([833, 675]), np.array([792, 677]), np.array([804, 744]), np.array([717, 760]), np.array([827, 833]), np.array([697, 827]), np.array([865, 899]), np.array([719, 909]), np.array([871, 943]), np.array([747, 927]), np.array([888, 931]), np.array([745, 919]), np.array([875, 909]), np.array([735, 905]), np.array([804, 923]), np.array([754, 917]), np.array([ 823, 1018]), np.array([ 756, 1020]), np.array([ 819, 1098]), np.array([ 756, 1100]), np.array([ 554, 1153]), np.array([ 615, 1171]), np.array([ 682, 1179]), np.array([ 806, 1189]), np.array([760, 748])]


hat_status = {

    'universe':{'path': "ench_body\hat.png",
          'anno_path': "ench_body\hat.csv",
          'morph': False, 'animated': False, 'has_alpha': True},
}

hair_status = {
    'universe':{'path': "ench_body\hair.png",
          'anno_path': "ench_body\hair.csv",
          'morph': False, 'animated': False, 'has_alpha': True},

}
shadow_status = {

    'universe':{'path': "ench_body\shadow.png",
          'anno_path': "ench_body\shadow.csv",
          'morph': False, 'animated': False, 'has_alpha': True},

}
upperarm_status = {

    'universe':{'path': "ench_body\shadow.png",
          'anno_path': "ench_body\shadow.csv",
          'morph': False, 'animated': False, 'has_alpha': True},

}

left_big_arm_status = {

    'universe':{'path': "ench_body\LArm_s2.png",
          'anno_path': "ench_body\LArm_s2.csv",
          'morph': False, 'animated': False, 'has_alpha': True},

}

right_big_arm_status = {

    'universe':{'path': "ench_body\RArm_s2.png",
          'anno_path': "ench_body\RArm_s2.csv",
          'morph': False, 'animated': False, 'has_alpha': True},

}

left_small_arm_status = {

    'universe':{'path': "ench_body\LForearm_s2.png",
          'anno_path': "ench_body\LForearm_s2.csv",
          'morph': False, 'animated': False, 'has_alpha': True},

}

right_small_arm_status = {
    'universe':{'path': "ench_body\RForearm_s2.png",
          'anno_path': "ench_body\RForearm_s2.csv",
          'morph': False, 'animated': False, 'has_alpha': True},
    
}

left_big_tui_status = {

    'universe':{'path': "ench_body\left_big_tui.png",
          'anno_path': "ench_body\left_big_tui.csv",
          'morph': False, 'animated': False, 'has_alpha': True},

}

left_small_tui_status = {

    'universe':{'path': "ench_body\left_small_tui.png",
          'anno_path': "ench_body\left_small_tui.csv",
          'morph': False, 'animated': False, 'has_alpha': True},

}

right_big_tui_status = {

    'universe':{'path': r"ench_body\right_big_tui.png",
          'anno_path': r"ench_body\right_big_tui.csv",
          'morph': False, 'animated': False, 'has_alpha': True},

}

right_small_tui_status = {

    'universe':{'path': r"ench_body\right_small_tui.png",
          'anno_path': r"ench_body\right_small_tui.csv",
          'morph': False, 'animated': False, 'has_alpha': True},

}

weapon_status = {

    'universe':{'path': "ench_body\MainWeapon_s2.png",
          'anno_path': "ench_body\MainWeapon_s2.csv",
          'morph': False, 'animated': False, 'has_alpha': True},
    
}

static_head_status = {

    'universe':{'path': "ench_body\static_head.png",
          'anno_path': "ench_body\static_head.csv",
          'morph': False, 'animated': False, 'has_alpha': True},
    
}
body_status = {

    'universe':{'path': "ench_body\\body.png",
          'anno_path': "ench_body\\body.csv",
          'morph': False, 'animated': False, 'has_alpha': True},
}

ear_status = {

    'universe':{'path': "ench\ear.png",
          'anno_path': "ench\ear.csv",
          'morph': False, 'animated': False, 'has_alpha': True},
    
}


head_status = {

    'universe':{'path': "ench\head.png",
          'anno_path': "ench\head.csv",
          'morph': False, 'animated': False, 'has_alpha': True},

    'universe_closed':{'path': "ench\head2.png",
          'anno_path': "ench\head.csv",
          'morph': False, 'animated': False, 'has_alpha': True},
          
    'universe_smile':{'path': "ench\head3.png",
          'anno_path': "ench\head.csv",
          'morph': False, 'animated': False, 'has_alpha': True},
    
    'universe_closed_smile':{'path': "ench_copy\head4.png",
          'anno_path': "ench\head.csv",
          'morph': False, 'animated': False, 'has_alpha': True},

    

    'left_look':{'path': "ench\TorsoTop_s2.png",
          'anno_path': "ench\TorsoTop_s2.csv",
          'morph': False, 'animated': False, 'has_alpha': True},
    
    'right_look':{'path': "ench\TorsoTop_s2.png",
          'anno_path': "ench\TorsoTop_s2.csv",
          'morph': False, 'animated': False, 'has_alpha': True},
    
}

head_hair_status = {
    'universe':{'path': "ench\Head_1_s2.png",
          'anno_path': "ench\Head_1_s2.csv",
          'morph': False, 'animated': False, 'has_alpha': True},
}

helm_status = {
    'universe':{'path': "ench\Helm_s2.png",
          'anno_path': "ench\Helm_s2.csv",
          'morph': False, 'animated': False, 'has_alpha': True},
}

eyes_box_status = {


    'universe':{'path': "ench\eyes_box.png",
          'anno_path': "ench\eye_o.csv",
          'morph': False, 'animated': False, 'has_alpha': True},


}

eyes_status = {

    'universe':{'path': "ench\eye_o.png",
          'anno_path': "ench\eye_o.csv",
          'morph': False, 'animated': False, 'has_alpha': True},

    'closed':{'path': "ench\Eyes_closed.png",
          'anno_path': "ench\Eyes_closed.csv",
          'morph': False, 'animated': False, 'has_alpha': True},
    
}

mouse_status = {

    'open':{'path': "ench\mouse.png",
          'anno_path': "ench\mouse.csv",
          'morph': False, 'animated': False, 'has_alpha': True},
    
}
'''
filters_config = {
    
    'ench':
        [
            {'path': "ench\TorsoTop_s2.png",
          'anno_path': "ench\TorsoTop_s2.csv",
          'morph': False, 'animated': False, 'has_alpha': True},

          {'path': "ench\Head_1_s2.png",
          'anno_path': "ench\Head_1_s2.csv",
          'morph': False, 'animated': False, 'has_alpha': True},

          {'path': "ench\REar_s2.png",
          'anno_path': "ench\REar_s2.csv",
          'morph': False, 'animated': False, 'has_alpha': True},

          {'path': "ench\LEar_s2.png",
          'anno_path': "ench\LEar_s2.csv",
          'morph': False, 'animated': False, 'has_alpha': True},



            {'path': "ench\Head_s2.png",
          'anno_path': "ench\Head_s2.csv",
          'morph': False, 'animated': False, 'has_alpha': True},


          {'path': "ench\Helm_s2.png",
          'anno_path': "ench\Helm_s2.csv",
          'morph': False, 'animated': False, 'has_alpha': True},
            
            {'path': "ench\Eyes.png",
          'anno_path': "ench\Eyes.csv",
          'morph': False, 'animated': False, 'has_alpha': True}
          
        ]
}
'''

def get_filename(video_path):
    file_without_extension = os.path.splitext(video_path)[0]
    return file_without_extension
#detect pose landmarks in image

def get_pos_landmarks(img):

    mp_pose = mp.solutions.pose
    height, width = img.shape[:-1]

    #pose = mp_pose.Pose()
    with mp_pose.Pose(static_image_mode=True, model_complexity=0, smooth_landmarks=True) as pose:

        
        frame_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # 运行姿势估计模型
        results = pose.process(frame_rgb)

        if results.pose_world_landmarks:

            # 处理姿势估计结果
            landmarks = []
            for landmark in results.pose_landmarks.landmark:
                landmarks.append([
                    landmark.x,
                    landmark.y
                    
                ])
            landmarks = np.array(landmarks)
            landmarks = landmarks*(width,height)

            landmarks = landmarks.astype('int')

            relevant_keypnts = []

            for i in range(len(landmarks)):
                relevant_keypnts.append(landmarks[i])
            

            return relevant_keypnts

        return 0


# detect facial landmarks in image
def getLandmarks(img):
    #mp.solutions.face_mesh 模块专门用于检测人脸的面部关键点
    mp_face_mesh = mp.solutions.face_mesh
    selected_keypoint_indices = [127, 93, 58, 136, 150, 149, 176, 148, 152, 377, 400, 378, 379, 365, 288, 323, 356, 70, 63, 105, 66, 55,
                 285, 296, 334, 293, 300, 168, 6, 195, 4, 64, 60, 94, 290, 439, 33, 160, 158, 173, 153, 144, 398, 385,
                 387, 466, 373, 380, 61, 40, 39, 0, 269, 270, 291, 321, 405, 17, 181, 91, 78, 81, 13, 311, 306, 402, 14,
                 178, 162, 54, 67, 10, 297, 284, 389,468,473]

    height, width = img.shape[:-1] #切片写法，不包含通道数

    # with 代码块结束时自动清理资源
    with mp_face_mesh.FaceMesh(max_num_faces=1, static_image_mode=True, min_detection_confidence=0.5,refine_landmarks = True) as face_mesh:


        '''
        MediaPipe 库通常使用RGB颜色空间来处理图像，所以在将图像传递给 face_mesh.process 函数之前，需要确保图像的颜色空间与库的要求相匹配。如果不进行转换，可能会导致颜色通道的混淆或不正确的结果。
        
        '''
        results = face_mesh.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)) #BGR颜色空间的图像转换为RGB颜色空间

        if not results.multi_face_landmarks:
            print('Face not detected!!!')
            return 0



        for face_landmarks in results.multi_face_landmarks:

            values = np.array(face_landmarks.landmark)
            face_keypnts = np.zeros((len(values), 2)) #

            for idx,value in enumerate(values):
                face_keypnts[idx][0] = value.x
                face_keypnts[idx][1] = value.y

            # Convert normalized points to image coordinates
            face_keypnts = face_keypnts * (width, height) #NumPy 数组的广播（broadcasting）功能来将所有坐标点同时乘以 (width, height)

            '''
            import numpy as np

            a = np.array([[0.1,0.6],[0.5,1.7]])

            a = a.astype('int')

            print(a)

            '''
            face_keypnts = face_keypnts.astype('int')

            relevant_keypnts = []

            for i in selected_keypoint_indices:
                relevant_keypnts.append(face_keypnts[i])
            return relevant_keypnts
    return 0


def load_filter_img(img_path, has_alpha):
    # Read the image
    img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)

    alpha = None
    if has_alpha:
        b, g, r, alpha = cv2.split(img)
        #后续叠加图像，避免透明度信息的影响，将alpha分离出来
        img = cv2.merge((b, g, r))

    return img, alpha


# annotation_file: csv file
def load_landmarks(annotation_file):
    with open(annotation_file) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=",") #指定分隔符为逗号（,）
        points = {}
        for i, row in enumerate(csv_reader):
            # skip head or empty line if it's there
            try:
                x, y = int(row[1]), int(row[2])
                points[row[0]] = (x, y)
            except ValueError:
                continue
        '''
        points eg
        {'0': (422, 73), '15': (760, 79)}

        '''
        return points


#双点映射不需要计算凸包
def find_convex_hull(points):
    hull = []
    '''
        a = {'0': (422, 73), '15': (760, 79)}

        print(np.array(list(a.values())))

        out: [[422  73]
            [760  79]]

    '''
    hullIndex = cv2.convexHull(np.array(list(points.values())), clockwise=False, returnPoints=False)
    addPoints = [
        [48], [49], [50], [51], [52], [53], [54], [55], [56], [57], [58], [59],  # Outer lips
        [60], [61], [62], [63], [64], [65], [66], [67],  # Inner lips
        [27], [28], [29], [30], [31], [32], [33], [34], [35],  # Nose
        [36], [37], [38], [39], [40], [41], [42], [43], [44], [45], [46], [47],  # Eyes
        [17], [18], [19], [20], [21], [22], [23], [24], [25], [26]  # Eyebrows
    ]
    hullIndex = np.concatenate((hullIndex, addPoints))
    for i in range(0, len(hullIndex)):
        hull.append(points[str(hullIndex[i][0])])

    return hull, hullIndex


def load_filter(filter_name="dog",filters_config= 0):

    filters = filters_config[filter_name]

    multi_filter_runtime = []

    for filter in filters:
        temp_dict = {}

        img1, img1_alpha = load_filter_img(filter['path'], filter['has_alpha'])

        temp_dict['img'] = img1
        temp_dict['img_a'] = img1_alpha

        points = load_landmarks(filter['anno_path'])

        temp_dict['points'] = points

        if filter['morph']:
            # Find convex hull for delaunay triangulation using the landmark points
            hull, hullIndex = find_convex_hull(points)

            # Find Delaunay triangulation for convex hull points
            sizeImg1 = img1.shape
            rect = (0, 0, sizeImg1[1], sizeImg1[0])
            dt = fbc.calculateDelaunayTriangles(rect, hull)

            temp_dict['hull'] = hull
            temp_dict['hullIndex'] = hullIndex
            temp_dict['dt'] = dt

            if len(dt) == 0:
                continue

        if filter['animated']:
            filter_cap = cv2.VideoCapture(filter['path'])
            temp_dict['cap'] = filter_cap

        multi_filter_runtime.append(temp_dict)

     

    return filters, multi_filter_runtime


def get_time():

    now = datetime.datetime.now()
    ts = now.timestamp()
    otherStyleTime = now.strftime("%Y-%m-%d-%H-%M-%S")
    return otherStyleTime

    
    

def process_video(video_path):
    # process input from webcam or video file
    cap = cv2.VideoCapture(video_path)

    frames_list = []
    points_list = []

    real_frame_index = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        else:
            temp_frame = frame
            points3 = get_pos_landmarks(temp_frame)
            if not points3 or (len(points3) <33):
                print("检测点不够")
                continue
            else:
                print("插入关键点"+str(real_frame_index)+"........")
                frames_list.append(temp_frame)
                points_list.append(points3)
                real_frame_index = real_frame_index+1
            

    #points_list = get_filtered_points(points_list)



    # Some variables
    count = 0
    isFirstFrame = True
    sigma = 50


    # 加载Haar级联分类器,用来识别眼部是否开闭
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

    #存储视频所需全局变量
    output_file = get_filename(video_path)
    output_file = output_file + get_time()+".mp4"
    print(output_file)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 使用MP4V编解码器
    frame_rate = 30 #目标视频帧率

    
    
    frame_index = -1


    shadow_y = 0 #控制地平线高度
    root_y = 0 #控制肩膀高度

    frame0_points3 = None
    speed_ratio = 0.3 # speed_ratio在0到1之间，越接近1速度越慢

    queue = [] #缓冲帧队列

    delay_index = [35,36,37,38] #需要延迟传递的坐标索引


    posRetarget = PosRetarget('rule_points.json')

    reference_vector = None
    
    #---------------------------------------------------------------------------------------------------------------
    blg = cv2.imread("blg.jpg")
    height, width, layers = blg.shape
    out = cv2.VideoWriter(output_file, fourcc, frame_rate, (width, height))
    target_point = get_target_point_frome_blg(blg)

    pre_point3 = []

    distance_thresold = []

    #---------------------------------------------------------------------------------------------------------------
    # The main loop
    for i in range(len(points_list)):

        if True:
            
        

            #---------------------------身体部分start----------------------------------------------------------------------------------------------------------
            '''
            points2 = getLandmarks(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            if points2 != 0 and (len(points2) == 77):

                # 灰度转换
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                # 检测眼睛
                eyes = eye_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

                #检测嘴巴开闭

                points = np.array([points2[61],points2[63],points2[65],points2[67]])

                area = calculate_quadrilateral_area(points)
                pass

            '''

            points3 = points_list[i]
            frame_index = frame_index+1
            print(frame_index)


            #----自适应滤波----------------------------------------------------------------------------------------------------------------------------
            
            if frame_index == 0:
                pre_point3 = points3
            
            else:
                distance = np.linalg.norm(points3[11] - pre_point3[11])
                distance_thresold.append(distance)
                distance_thresold.sort()

                
                half_index = math.ceil(len(distance_thresold)/2) -1

                ts = distance_thresold[half_index]
                
                print(ts)

                points3 = adaptive_filter(pre_point3,points3,threshold=2,alpha=0.1)
                pre_point3 = points3

            
            #------------------------------------------------------------------------------------------------------------------------------------------
            '''
            
            ################ Optical Flow and Stabilization Code #####################-------------------------------------------------------------------------------
            img2Gray = cv2.cvtColor(frames_list[i], cv2.COLOR_BGR2GRAY)


            
            if isFirstFrame:
                points2Prev = np.array(points3, np.float32)
                img2GrayPrev = np.copy(img2Gray) #为光流法计算作准备
                isFirstFrame = False
            #Lucas-Kanade光流法（Optical Flow）的参数 lk_params
            lk_params = dict(winSize=(101, 101), maxLevel=15,
                            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.001))
            
            #my_dict = dict(a=1, b=2, c=3)
            #结果：{'a': 1, 'b': 2, 'c': 3}
            points2Next, st, err = cv2.calcOpticalFlowPyrLK(img2GrayPrev, img2Gray, points2Prev,
                                                            np.array(points3, np.float32),
                                                            **lk_params) #lk_params参数解包

            # Final landmark points are a weighted average of detected landmarks and tracked landmarks

            for k in range(0, len(points3)):
                #计算真实帧和光流计算帧的距离
                d = cv2.norm(np.array(points3[k]) - points2Next[k])
                
                alpha = math.exp(-d * d / sigma)
                points3[k] = (1 - alpha) * np.array(points3[k]) + alpha * points2Next[k]
                points3[k] = fbc.constrainPoint(points3[k], frames_list[i].shape[1], frames_list[i].shape[0])
                points3[k] = np.array([int(points3[k][0]), int(points3[k][1])])

            # Update variables for next pass
            points2Prev = np.array(points3, np.float32)
            img2GrayPrev = img2Gray
            ################ End of Optical Flow and Stabilization Code ###############-----------------------------------------------------------------------------

            '''
            


            frame = blg

            
            
            if frame_index == 0:
                frame0_points3 = points3
                print("dadafafaefaefsaefaef")
                print(frame0_points3)

            
            speed_vector = points3[11] - frame0_points3[11]
            speed_vector = speed_ratio*speed_vector

            for i in range(len(points3)):
                    #取负回移
                points3[i] = points3[i] - speed_vector

            
            
            if frame_index == 0:
                root_y = points3[11][1]
            
            
            if points3[11][1]<root_y:
                points3[11][1] = root_y
            
            
            posRetarget.frame_points = points3

            posRetarget.pos_fresh()

            points3 = posRetarget.frame_points
            
            #特殊点延迟传递
            queue.append(copy.deepcopy(points3))
            if len(queue)>=5:

                first_points = queue.pop(0)
                #从配置数据中获取要延迟传递的坐标
                for i in range(len(delay_index)):
                    index = delay_index[i]
                    points3[index] = first_points[index]

            #y阴影水平矫正
            if frame_index == 0:

                shadow_y = points3[33][1]

            points3[33][1] = shadow_y
            points3[34][1] = shadow_y

            

            #points3再次重映射--------------------------------------------------------------------------------------------------------

            if frame_index == 0:
                reference_vector = target_point - points3[34]
            
            last_points3 = rule_points(reference_vector,points=points3)

            points3 = last_points3


            
            
            #points3[10][1] = points3[9][1]
            filters_config = {}

            filters_config['ench'] = []
            #insert body
            
            

            filters_config['ench'].append(left_big_arm_status['universe'])
            filters_config['ench'].append(left_small_arm_status['universe'])

            filters_config['ench'].append(shadow_status["universe"])
            
            
            filters_config['ench'].append(left_big_tui_status["universe"])
            filters_config['ench'].append(left_small_tui_status["universe"])

            
            filters_config['ench'].append(right_big_tui_status["universe"])
            filters_config['ench'].append(right_small_tui_status["universe"])
            filters_config['ench'].append(body_status['universe'])

            

            filters_config['ench'].append(hair_status['universe'])
            
            #filters_config['ench'].append(weapon_status['universe'])

            
            #insert static head
            filters_config['ench'].append(static_head_status['universe'])
            
            filters_config['ench'].append(hat_status['universe'])
            filters_config['ench'].append(right_big_arm_status['universe'])
            filters_config['ench'].append(right_small_arm_status['universe'])
            iter_filter_keys = iter(filters_config.keys())
            filters, multi_filter_runtime = load_filter(next(iter_filter_keys),filters_config)
            
            
            for idx, filter in enumerate(filters):

                filter_runtime = multi_filter_runtime[idx]
                img1 = filter_runtime['img']
                points1 = filter_runtime['points']
                img1_alpha = filter_runtime['img_a']

                if filter['morph']:

                    hullIndex = filter_runtime['hullIndex']
                    dt = filter_runtime['dt']
                    hull1 = filter_runtime['hull']

                    # create copy of frame
                    warped_img = np.copy(frame)

                    # Find convex hull
                    hull2 = []
                    for i in range(0, len(hullIndex)):
                        hull2.append(points2[hullIndex[i][0]])

                    mask1 = np.zeros((warped_img.shape[0], warped_img.shape[1]), dtype=np.float32)
                    mask1 = cv2.merge((mask1, mask1, mask1))
                    img1_alpha_mask = cv2.merge((img1_alpha, img1_alpha, img1_alpha))

                    # Warp the triangles
                    for i in range(0, len(dt)):
                        t1 = []
                        t2 = []

                        for j in range(0, 3):
                            t1.append(hull1[dt[i][j]])
                            t2.append(hull2[dt[i][j]])

                        fbc.warpTriangle(img1, warped_img, t1, t2)
                        fbc.warpTriangle(img1_alpha_mask, mask1, t1, t2)

                    # Blur the mask before blending
                    mask1 = cv2.GaussianBlur(mask1, (3, 3), 10)

                    mask2 = (255.0, 255.0, 255.0) - mask1

                    # Perform alpha blending of the two images
                    temp1 = np.multiply(warped_img, (mask1 * (1.0 / 255)))
                    temp2 = np.multiply(frame, (mask2 * (1.0 / 255)))

                    output = temp1 + temp2
                else:
                    #dst_points = [points3[int(list(points1.keys())[0])], points3[int(list(points1.keys())[1])]]
                    
                    dst_points = []
                    for i in range(len(list(points1.keys()))):
                        dst_points.append(points3[int(list(points1.keys())[i])])
                        
                    
                    tform = fbc.similarityTransform(list(points1.values()), dst_points)
                    # Apply similarity transform to input image
                    trans_img = cv2.warpAffine(img1, tform, (frame.shape[1], frame.shape[0]))
                    trans_alpha = cv2.warpAffine(img1_alpha, tform, (frame.shape[1], frame.shape[0]))
                    mask1 = cv2.merge((trans_alpha, trans_alpha, trans_alpha))

                    # Blur the mask before blending
                    mask1 = cv2.GaussianBlur(mask1, (3, 3), 10)

                    mask2 = (255.0, 255.0, 255.0) - mask1

                    # Perform alpha blending of the two images
                    temp1 = np.multiply(trans_img, (mask1 * (1.0 / 255)))
                    temp2 = np.multiply(frame, (mask2 * (1.0 / 255)))

                    output = temp1 + temp2

                frame = output = np.uint8(output)

            if True:
                #cv2.imshow("Face Filter", output)
                
                out.write(output)

                #print("写入。。。。。。。。。。。。。。。。。。。。。。")
                keypressed = cv2.waitKey(1) & 0xFF
                if keypressed == ord("q"):
                    break
                continue
            continue

            
            #--------------------------脸部部分start------------------------------------------------------------------------------------------------------------

            # if face is partially detected
            if not points2 or (len(points2) != 77):
                continue
            #-------------------------状态识别----------------------------------------------------


            y1 = calculate_distance(points2[28],points2[45])
            y2 = calculate_distance(points2[28],points2[36])


            if y1-y2<-25:
                print("左转")
            elif y1-y2>25:
                print("右转")
            else:
                print("正脸")
            
            
            filters_config = {}

            filters_config['ench'] = []
            #初始化身体姿态

            #filters_config['ench'].append(upperarm_status['universe'])


            #filters_config['ench'].append(body_status['universe'])
            

            filters_config['ench'].append(ear_status['universe'])
            
            filters_config['ench'].append(head_hair_status['universe'])


            
            if len(eyes) <= 0 and area<1000:
                filters_config['ench'].append(head_status['universe_closed'])
                
                filters_config['ench'].append(eyes_status['closed'])
                
                
            elif len(eyes) <= 0 and area>=1000:
                
                
                filters_config['ench'].append(head_status['universe_closed_smile'])
                
                filters_config['ench'].append(eyes_status['closed'])

            elif len(eyes) > 0 and area<1000:
                print("睁眼")
                
                filters_config['ench'].append(head_status['universe'])
                filters_config['ench'].append(eyes_status['universe'])
            
            elif len(eyes) > 0 and area>=1000:
                print("睁眼")
                print("张嘴")
                
                filters_config['ench'].append(head_status['universe_smile'])
                filters_config['ench'].append(eyes_status['universe'])
            
                
                

            iter_filter_keys = iter(filters_config.keys())
            filters, multi_filter_runtime = load_filter(next(iter_filter_keys),filters_config)

            #-----------------------状态识别结束-------------------------------------------------------

        
            

            if VISUALIZE_FACE_POINTS:
                for idx, point in enumerate(points2):
                    cv2.circle(frame, point, 2, (255, 0, 0), -1)
                    cv2.putText(frame, str(idx), point, cv2.FONT_HERSHEY_SIMPLEX, .3, (255, 255, 255), 1)
                cv2.imshow("landmarks", frame)

            '''
            green_blg = create_green_blg(frame.shape[0],frame.shape[1])
            frame = green_blg
            '''

            for idx, filter in enumerate(filters):

                filter_runtime = multi_filter_runtime[idx]
                img1 = filter_runtime['img']
                points1 = filter_runtime['points']
                img1_alpha = filter_runtime['img_a']

                if filter['morph']:

                    hullIndex = filter_runtime['hullIndex']
                    dt = filter_runtime['dt']
                    hull1 = filter_runtime['hull']

                    # create copy of frame
                    warped_img = np.copy(frame)

                    # Find convex hull
                    hull2 = []
                    for i in range(0, len(hullIndex)):
                        hull2.append(points2[hullIndex[i][0]])

                    mask1 = np.zeros((warped_img.shape[0], warped_img.shape[1]), dtype=np.float32)
                    mask1 = cv2.merge((mask1, mask1, mask1))
                    img1_alpha_mask = cv2.merge((img1_alpha, img1_alpha, img1_alpha))

                    # Warp the triangles
                    for i in range(0, len(dt)):
                        t1 = []
                        t2 = []

                        for j in range(0, 3):
                            t1.append(hull1[dt[i][j]])
                            t2.append(hull2[dt[i][j]])

                        fbc.warpTriangle(img1, warped_img, t1, t2)
                        fbc.warpTriangle(img1_alpha_mask, mask1, t1, t2)

                    # Blur the mask before blending
                    mask1 = cv2.GaussianBlur(mask1, (3, 3), 10)

                    mask2 = (255.0, 255.0, 255.0) - mask1

                    # Perform alpha blending of the two images
                    temp1 = np.multiply(warped_img, (mask1 * (1.0 / 255)))
                    temp2 = np.multiply(frame, (mask2 * (1.0 / 255)))

                    output = temp1 + temp2
                else:
                    #dst_points = [points2[int(list(points1.keys())[0])], points2[int(list(points1.keys())[1])]]
                    dst_points = []
                    for i in range(len(list(points1.keys()))):
                        dst_points.append(points2[int(list(points1.keys())[i])])
                    

                                   
                    tform = fbc.similarityTransform(list(points1.values()), dst_points)
                    # Apply similarity transform to input image
                    trans_img = cv2.warpAffine(img1, tform, (frame.shape[1], frame.shape[0]))
                    trans_alpha = cv2.warpAffine(img1_alpha, tform, (frame.shape[1], frame.shape[0]))
                    mask1 = cv2.merge((trans_alpha, trans_alpha, trans_alpha))

                    # Blur the mask before blending
                    mask1 = cv2.GaussianBlur(mask1, (3, 3), 10)

                    mask2 = (255.0, 255.0, 255.0) - mask1

                    # Perform alpha blending of the two images
                    temp1 = np.multiply(trans_img, (mask1 * (1.0 / 255)))
                    temp2 = np.multiply(frame, (mask2 * (1.0 / 255)))

                    output = temp1 + temp2

                frame = output = np.uint8(output)

            #cv2.putText(frame, "Press F to change filters", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 0, 0), 1)

            #cv2.imshow("Face Filter", output)
            if frame_index%2 == 0:
                    out.write(output)
            keypressed = cv2.waitKey(1) & 0xFF
            if keypressed == ord("q"):
                break
            # Put next filter if 'f' is pressed
            elif keypressed == ord('f'):
                try:
                    filters, multi_filter_runtime = load_filter(next(iter_filter_keys),filters_config)
                except:
                    iter_filter_keys = iter(filters_config.keys())
                    filters, multi_filter_runtime = load_filter(next(iter_filter_keys),filters_config)
                    

            count += 1

    cap.release()
    out.release()
    cv2.destroyAllWindows()

    return output_file


sem = Semaphore(5)
from flask import render_template
@app.route('/')
def index():
    # 返回一个名为index.html的模板文件
    return render_template('index.html')



@app.route('/upload', methods=['POST'])
def upload():

    if not sem.acquire(blocking=False):
        return 400

    # 获取上传的文件对象
    file = request.files['video']

    sem.release()
    # 检查文件是否为视频格式
    if file and file.content_type.startswith('video/'):
        # 定义一个文件夹路径，用于保存视频
        folder = 'videos'
        # 获取文件名
        filename = file.filename
        
        filename = get_time()+filename
        # 拼接完整的文件路径
        filepath = os.path.join(folder, filename)
        # 保存文件到指定文件夹
        file.save(filepath)
        # 调用一个函数，用于处理视频
        processed_video = process_video(filepath)
    
        # 返回处理后的视频给前端
        return send_file(processed_video, mimetype='video/mp4')
    else:
        # 如果文件不是视频格式，返回一个错误信息
        return 'Invalid video file', 400
    


if __name__ == '__main__':
    #app.run(host='0.0.0.0', port=8000)
    video_path = sys.argv[1]
    process_video(video_path)