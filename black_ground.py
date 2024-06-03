import cv2
class Blg:
    def __init__(self,blg_video_path):
        self.blg_video_path = blg_video_path
        self.blg_frames = []

        pass

    def create_frames(self):

        cap = cv2.VideoCapture(self.blg_video_path)
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            else:
                self.blg_frames.append(frame)
        
        cap.release()
        return self.blg_frames
        
