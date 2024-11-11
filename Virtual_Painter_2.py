import cv2
import numpy as np
from cvzone.HandTrackingModule import HandDetector
import math

(width_c,height_c) = 1000,750

cap = cv2.VideoCapture(0)
cap.set(3,width_c)
cap.set(4,height_c)

detector = HandDetector(maxHands=1)

Result_img = None
Black_board = np.zeros((height_c,width_c,3),dtype=np.uint8)

prev_pos = None
color_index = 0
brush_thickness = 10
selected_mode = 1

colors = {
    "Red": (0,0,255),
    "Blue": (255,0,0),
    "Green": (0,255,0),
    "White": (255,255,255),
    "Black": (0,0,0)
}

color = [
    "Red",
    "Blue",
    "Green",
    "White",
    "Black"
]

def Draw(img,canvas,hand_info,prev_pos,color,thickness):
    hands,img = hand_info.findHands(img,draw=False,flipType=False)
    current_pos = None
    if hands:
        hand = hands[0]
        x1, y1 = hand["lmList"][4][:2]
        x2, y2 = hand["lmList"][8][:2]
        cx, cy = (x2 + x1) // 2, (y2 + y1) // 2
        distance = math.hypot(x2 - x1, y2 - y1)
        #print(distance)
        if int(distance) < 57:
            current_pos = (cx,cy)
            if prev_pos is None: prev_pos = current_pos
            cv2.line(canvas,current_pos,prev_pos,color,thickness)
        else:
            cv2.line(img,(x1,y1),(x2,y2),(255,255,255),4)
            cv2.circle(img,(x1,y1),10,(0,0,255),-1)
            cv2.circle(img,(x1,y1),10 + 5,(0,0,255),2)
            cv2.circle(img, (x2, y2), 10, (0, 0, 255), -1)
            cv2.circle(img, (x2, y2), 10 + 5, (0, 0, 255), 2)
            cv2.circle(img,(cx,cy),10 + 5,(255,255,255),-1)
        cv2.circle(img,(cx,cy),10,color,-1)
    return current_pos

while True:
    success,Original_img = cap.read()
    Original_img = cv2.resize(Original_img,(width_c,height_c))
    Original_img = cv2.flip(Original_img,1)
    if not success:
        break
    else:
        prev_pos = Draw(Original_img,Black_board,detector,prev_pos,colors[color[color_index]],
                        brush_thickness)
        if selected_mode == 1:
            Result_img = cv2.addWeighted(Original_img,1,Black_board,0.5,0)
        elif selected_mode == 4:
            Result_img = cv2.bitwise_and(Original_img,Black_board)
        elif selected_mode == 2:
            Result_img = cv2.bitwise_or(Original_img,Black_board)
        elif selected_mode == 3:
            Result_img = cv2.bitwise_xor(Original_img,Black_board)
        cv2.putText(Result_img,f"Brush Thickness: {brush_thickness}",(15,height_c - 15),
                    cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),2)
        cv2.putText(Result_img,f"Selected Mode: {selected_mode}",(15,height_c-15 - 30),
                    cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),2)
        cv2.putText(Result_img,f"Color: {color[color_index]}",(15,height_c-15-30 - 30),
                    cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),2)
        cv2.imshow("Result Img",Result_img)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    elif key == ord("c"):
        Black_board = np.zeros((height_c,width_c,3),dtype=np.uint8)
    elif key == ord("r"):
        Black_board = np.zeros((height_c,width_c,3),dtype=np.uint8)
        color_index = 0
        brush_thickness = 10
        selected_mode = 1
    elif key == ord("a"):
        if color_index > 0:
            color_index -= 1
        else: color_index = len(color) - 1
    elif key == ord("d"):
        if color_index < len(color) - 1:
            color_index += 1
        else: color_index = 0
    elif key == ord("w"):
        if brush_thickness < 300:
            brush_thickness += 1
        else: brush_thickness = 1
    elif key == ord("s"):
        if brush_thickness > 1:
            brush_thickness -= 1
        else: brush_thickness = 300
    elif key == ord(" "):
        if selected_mode < 4:
            selected_mode += 1
        else: selected_mode = 1
cap.release()
cv2.destroyAllWindows()
