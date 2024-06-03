import cv2
import numpy as np

cv2.namedWindow("dadwa",cv2.WINDOW_NORMAL)
img = cv2.imread("ench_body\static_head.png")

frame = cv2.imread("20230922-154906.jpg")
h,w = frame.shape[:2]



pts1 = np.float32([[363,668],[811,691],[585,906],[668,908]])
pts2 = np.float32([[339,950],[770,956],[485,1305],[632,1302]])




M = cv2.getPerspectiveTransform(pts1,pts2)

result = cv2.warpPerspective(img,M,(w,h)) + frame

cv2.imshow("dadwa",result)
key = cv2.waitKey(0)
if key == ord("q"):
    cv2.destroyAllWindows()