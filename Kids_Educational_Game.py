import cv2
from cvzone.FaceMeshModule import FaceMeshDetector
from Utilities.Common_Utils import EuclideanDistance
from Utilities.Drawing_Utils import Ray
from random import randint

(width_c,height_c) = 1000,750

cap = cv2.VideoCapture(0)
cap.set(3,width_c)
cap.set(4,height_c)

detector = FaceMeshDetector(maxFaces=1)

ids_list = [0,17,61,291]
mouth = None
score = 0
game_over = False
eatable_colors = [(0,255,0),(255,0,0),(255,255,0)]
non_eatable_colors = [(0,0,255),(255,0,255),(0,255,255)]
speed = 10
food_radius = randint(20,50)
food_x = randint(food_radius,width_c - food_radius)
food_y = food_radius

if randint(1, 2) == 1:
    color = eatable_colors[randint(1, len(eatable_colors) - 1)]
else:
    color = non_eatable_colors[randint(1, len(non_eatable_colors) - 1)]

def Game(img,mouth,face,id_list):
    global food_x,food_radius,food_y,score,eatable_colors,\
        non_eatable_colors,color,speed,game_over
    cx = (face[id_list[0]][0] + face[id_list[1]][0] + face[id_list[2]][0] + face[id_list[3]][0]) // 4
    cy = (face[id_list[0]][1] + face[id_list[1]][1] + face[id_list[2]][1] + face[id_list[3]][1]) // 4
    cv2.circle(img,(food_x,food_y),food_radius,color,-1)
    Ray(img,(cx,cy),(food_x,food_y),(0,255,0),3,False)
    food_y += speed
    distance = EuclideanDistance((cx,cy),(food_x,food_y))
    if food_y > height_c - food_radius:
        food_radius = randint(20, 50)
        food_x = randint(food_radius, width_c - food_radius)
        food_y = food_radius
        if randint(1, 2) == 1:
            color = eatable_colors[randint(1, len(eatable_colors) - 1)]
        else:
            color = non_eatable_colors[randint(1, len(non_eatable_colors) - 1)]
    if mouth == "Open" and distance < 40 and color in eatable_colors:
        food_radius = randint(20, 50)
        food_x = randint(food_radius, width_c - food_radius)
        food_y = food_radius
        score += 1
        speed += 1
        if speed > 20:
            speed = 20
        if randint(1, 2) == 1:
            color = eatable_colors[randint(1, len(eatable_colors) - 1)]
        else:
            color = non_eatable_colors[randint(1, len(non_eatable_colors) - 1)]
    elif mouth == "Open" and distance < 40 and color in non_eatable_colors:
        food_radius = randint(20, 50)
        food_x = randint(food_radius, width_c - food_radius)
        food_y = food_radius
        score -= 2
        speed -= 1
        if speed < 10:
            speed = 10
        if randint(1, 2) == 1:
            color = eatable_colors[randint(1, len(eatable_colors) - 1)]
        else:
            color = non_eatable_colors[randint(1, len(non_eatable_colors) - 1)]
    if score < 0:
        game_over = True

while True:
    success,Original_img = cap.read()
    Original_img = cv2.resize(Original_img,(width_c,height_c))
    Original_img = cv2.flip(Original_img,1)
    if not success:
        break
    else:
        Original_img,faces = detector.findFaceMesh(Original_img,draw=False)
        if not game_over:
            if faces:
                for face in faces:
                    cv2.line(Original_img,face[ids_list[0]],face[ids_list[1]],(0,255,0),3)
                    cv2.line(Original_img,face[ids_list[2]],face[ids_list[3]],(0,255,0),3)
                    cv2.circle(Original_img,face[ids_list[0]],7,(0,0,255),-1)
                    cv2.circle(Original_img, face[ids_list[1]], 7, (0, 0, 255), -1)
                    cv2.circle(Original_img, face[ids_list[2]], 7, (0, 0, 255), -1)
                    cv2.circle(Original_img, face[ids_list[3]], 7, (0, 0, 255), -1)
                    left_right = EuclideanDistance(face[ids_list[0]],face[ids_list[1]])
                    up_down = EuclideanDistance(face[ids_list[2]],face[ids_list[3]])
                    ratio = up_down / left_right
                    print(f"left-right: {left_right}\nup-down: {up_down}\nratio: {ratio}")
                    Game(Original_img,mouth,face,ids_list)
                    if ratio < 1.6:
                        mouth = "Open"
                    else:
                        mouth = "Closed"
                    cv2.putText(Original_img,mouth,(25,60),cv2.FONT_HERSHEY_SIMPLEX,2,(0,0,255),3)
                    cv2.putText(Original_img,str(score),(width_c - 150,60),cv2.FONT_HERSHEY_SIMPLEX,2,(0,0,255),3)
        else:
            cv2.putText(Original_img,f"Game Over",(width_c // 2 - 275,height_c // 2),cv2.FONT_HERSHEY_SIMPLEX,3,(0,255,0),5)
        cv2.imshow("Original Img",Original_img)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    elif key == ord("r"):
        game_over = False
        food_radius = randint(20, 50)
        food_x = randint(food_radius, width_c - food_radius)
        food_y = food_radius
        if randint(1, 2) == 1:
            color = eatable_colors[randint(1, len(eatable_colors) - 1)]
        else:
            color = non_eatable_colors[randint(1, len(non_eatable_colors) - 1)]
        score = 0
        speed = 10
cap.release()
cv2.destroyAllWindows()
