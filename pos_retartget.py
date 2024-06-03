import numpy as np
import json
class PosRetarget:

    def __init__(self,rule_points_json,frame_points = None):

        self.frame_points = frame_points
        self.rule_points_json = rule_points_json
        self.rule_points = None
        self.ratio = 0.5 # how to caculate it?
        self.point_modify_list = [7,8,23,24,33,34,35,36,37,38,39,40]
        self.root_index = 11
        self.pos_index = [
            [11,13],
            [13,15],
            [11,23],
            [23,25],
            [25,27],
            [11,12],
            [12,14],
            [14,16],
            [12,24],
            [24,26],
            [26,28],
            [11,7],
            [11,8],
            [11,33],
            [11,34],
            [11,35],
            [11,36],
            [11,37],
            [11,38],
            [11,39],
            [11,40],
            
        ]

        self.rule_d = []

        self.get_rule_distance()
        #self.pos_fresh()
        pass
    
    def add_length(self):

        len1 = len(self.rule_points)
        len2 = len(self.frame_points)

        gap = len1 - len2

        for i in range(0,gap):
            self.frame_points.append(0)
    
    def get_rule_distance(self):

        with open(self.rule_points_json) as json_file:
            self.rule_points = json.load(json_file)
        
        for i in range(len(self.rule_points)):
            dict = self.rule_points[i]
            self.rule_points[i]  = np.array([dict['x'],dict['y']])
        
        for i in range(len(self.pos_index)):
            vector = self.rule_points[self.pos_index[i][1]]-self.rule_points[self.pos_index[i][0]]
            magnitude = np.linalg.norm(vector)
            self.rule_d.append(magnitude)
    
    def pos_fresh(self):
        self.add_length()
        norm_vectors = []
        for i in range(len(self.pos_index)):
            vector = self.frame_points[self.pos_index[i][1]] -self.frame_points[self.pos_index[i][0]]
            magnitude = np.linalg.norm(vector)
            # 计算单位向量
            unit_vector = vector / magnitude
            norm_vectors.append(unit_vector)

        for i in range(len(self.pos_index)):
            
            norm_vector = norm_vectors[i]
            d = self.rule_d[i]
            new_vector = self.ratio*d*norm_vector
            
            self.frame_points[self.pos_index[i][1]] = self.frame_points[self.pos_index[i][0]] + new_vector

        for item in self.point_modify_list:
            vector = self.rule_points[item] - self.rule_points[self.root_index]
            vector = self.ratio*vector
            self.frame_points[item] = vector + self.frame_points[self.root_index]
            
if __name__ == "__main__":


    with open('rule_points.json') as json_file:
        rule_points = json.load(json_file)

    for i in range(len(rule_points)):
            dict = rule_points[i]
            rule_points[i]  = np.array([dict['x'],dict['y']])

    print(rule_points)
    pass
        
            
            