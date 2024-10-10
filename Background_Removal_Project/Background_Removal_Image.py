import cv2
import numpy as np
from cvzone.SelfiSegmentationModule import SelfiSegmentation
import os

(width_c,height_c) = 700,500

cap = cv2.VideoCapture(0)
cap.set(3,width_c)
cap.set(4,height_c)

segmentor = SelfiSegmentation()

list_img = os.listdir("Background_Images")
img_list = []
for images in list_img:
    img = cv2.imread(f"Background_Images/{images}")
    img = cv2.resize(img,(width_c,height_c))
    img_list.append(img)

index_img = 0

while True:
    success,Original_img = cap.read()
    Original_img = cv2.resize(Original_img,(width_c,height_c))
    if not success:
        break
    else:
        Img_BG = segmentor.removeBG(Original_img, img_list[index_img], 0.8)

        Img_stacked = np.hstack((Original_img,Img_BG))
        cv2.imshow("Img Stacked",Img_stacked)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("a"):
        if index_img > 0:
            index_img -= 1
    elif key == ord("d"):
        if index_img < len(list_img) - 1:
            index_img += 1
    elif key == ord("q"):
        break
cap.release()
cv2.destroyAllWindows()
