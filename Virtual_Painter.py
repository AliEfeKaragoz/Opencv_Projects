import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import random

(width_c,height_c) = 1000,750

cap = cv2.VideoCapture(0)
cap.set(3,width_c)
cap.set(4,height_c)

Black_Board = np.zeros((height_c, width_c, 3),dtype=np.uint8)

hand_detector = HandDetector(maxHands=1,detectionCon=0.6)

index = 0
prev_pos = None
brush_thickness = 10
selected_mode = 1
b = random.randint(0,255)
g = random.randint(0,255)
r = random.randint(0,255)
random_color = (b,g,r)

colors = {
    "red": (0,0,255),
    "blue": (255,0,0),
    "green": (0,255,0),
    "yellow": (0,255,255),
    "purple": (255,0,255),
    "white": (255,255,255),
    "random": random_color,
    "black (eraser)": (0,0,0)
}

color = [
    "red",
    "blue",
    "green",
    "yellow",
    "purple",
    "white",
    "random",
    "black (eraser)"
]

def Draw(img,canvas,hand_info,prev_pos,color,thickness):
    hands,img = hand_info.findHands(img,draw=False)
    current_pos = None
    if hands:
        hand = hands[0]
        fingers = hand_info.fingersUp(hand)
        if fingers == [0,1,0,0,0]:
            current_pos = hand["lmList"][8][:2]
            cv2.circle(img,current_pos,10,color,-1)
            if prev_pos is None: prev_pos = current_pos
            cv2.line(canvas,current_pos,prev_pos,color,thickness)
    return current_pos

while True:
    success,Original_img = cap.read()
    Original_img = cv2.resize(Original_img,(width_c,height_c))
    Original_img = cv2.flip(Original_img,1)
    if not success:
        break
    else:
        prev_pos = Draw(Original_img,Black_Board,hand_detector,prev_pos,colors[color[index]],brush_thickness)
        Result_img = Original_img
        if selected_mode == 1:
            Result_img = cv2.addWeighted(Original_img, 1, Black_Board, 0.5, 0)
        elif selected_mode == 2:
            Result_img = cv2.bitwise_or(Original_img, Black_Board)
        elif selected_mode == 3:
            Result_img = cv2.bitwise_and(Original_img, Black_Board)
        elif selected_mode == 4:
            Result_img = cv2.bitwise_xor(Original_img, Black_Board)
        cv2.putText(Result_img,f"Color: {color[index]}",(25,35),cv2.FONT_HERSHEY_SIMPLEX,1.1,colors[color[index]],3)
        cv2.putText(Result_img,f"Brush Thickness: {brush_thickness}",(25,75),cv2.FONT_HERSHEY_SIMPLEX,1.1,colors["white"],3)
        cv2.putText(Result_img,f"Selected Mode: {selected_mode}",(25,height_c - 25),cv2.FONT_HERSHEY_SIMPLEX,1.1,colors["white"],3)
        cv2.imshow("Result Img",Result_img)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    elif key == ord("a"):
        if index > 0:
            index -= 1
    elif key == ord("d"):
        if index < len(color) - 1:
            index += 1
    elif key == ord("s"):
        if brush_thickness > 1:
            brush_thickness -= 1
    elif key == ord("w"):
        brush_thickness += 1000
    elif key == ord(" "):
        selected_mode += 1
        if selected_mode > 4:
            selected_mode = 1
    elif key == ord("c"):
        cv2.rectangle(Black_Board,(0,0),(width_c,height_c),(0,0,0),-1)
    elif key == ord("r"):
        cv2.rectangle(Black_Board, (0, 0), (width_c, height_c), (0, 0, 0), -1)
        index = 0
        brush_thickness = 10
        selected_mode = 1
cap.release()
cv2.destroyAllWindows()
