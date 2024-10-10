"""
Date: 08.24.2024

Status: Public

This file can be used in all projects.
"""

import cv2
import numpy as np
import math

def GetContours(img,canny_thr=[100,100],min_area=1000,filter_=0,show_canny=False,draw=False):
    Gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    Blur_img = cv2.GaussianBlur(Gray_img,(5,5),1)
    Canny_img = cv2.Canny(Blur_img,canny_thr[0],canny_thr[1])
    Canny_img = cv2.dilate(Canny_img,None,iterations=3)
    Canny_img = cv2.erode(Canny_img,None,iterations=2)
    if show_canny: cv2.imshow("Canny Img",Canny_img)
    contours,hierarchy = cv2.findContours(Canny_img,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    final_contours = []
    for i in contours:
        area = cv2.contourArea(i)
        if area > min_area:
            peri = cv2.arcLength(i,True)
            approx = cv2.approxPolyDP(i,0.02 * peri,True)
            bbox = cv2.boundingRect(approx)
            if filter_ > 0:
                if len(approx) == filter_:
                    final_contours.append([len(approx), area, approx, bbox, i])
    final_contours.sort(key=lambda x: x[1],reverse=True)
    if draw:
        for cont in final_contours:
            cv2.polylines(img,[cont[2]],True,(0,0,255),3)
    return final_contours

def ReorderRectanglePoints(points):
    new_points = np.zeros_like(points)
    points = points.reshape((4, 2))
    add = points.sum(1)
    new_points[0] = points[np.argmin(add)] # [0,0]
    new_points[3] = points[np.argmax(add)] # [w,h]
    diff = np.diff(points, axis=1)
    new_points[1] = points[np.argmin(diff)] # [w,0]
    new_points[2] = points[np.argmax(diff)] # [0,h]
    return new_points

def WarpImage(img,points,w,h,extract=0,inverse=False):
    points = ReorderRectanglePoints(points)
    pts1 = np.float32(points)
    pts2 = np.float32([[0,0],[w,0],[0,h],[w,h]])
    if inverse == False:
        M = cv2.getPerspectiveTransform(pts1,pts2)
        Warp_img = cv2.warpPerspective(img, M, (w, h))
        Warp_img = Warp_img[extract:Warp_img.shape[0] - extract, extract:Warp_img.shape[1] - extract]
        Warp_img = cv2.resize(Warp_img, (w, h))
    elif inverse == True:
        M = cv2.getPerspectiveTransform(pts2, pts1)
        Warp_img = cv2.warpPerspective(img, M, (w, h))
        Warp_img = Warp_img[extract:Warp_img.shape[0] - extract, extract:Warp_img.shape[1] - extract]
        Warp_img = cv2.resize(Warp_img, (w, h))
    return Warp_img

def EuclideanDistance(p1,p2):
    return math.sqrt((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2)

def FaceMeshIndexDictionary(index_group_name=str):
    face_mesh_index_dict = {
        "lips": [0, 267, 269, 270, 13, 14, 17, 402, 146, 405, 409, 415, 291,
                 37, 39, 40, 178, 308, 181, 310, 311, 312, 185, 314, 317, 318,
                 61, 191, 321, 324, 78, 80, 81, 82, 84, 87, 88, 91, 95, 375],
        "left_eye": [384, 385, 386, 387, 388, 390, 263, 362, 398, 466, 373, 374, 249, 380, 381, 382],
        "left_eyebrow": [293, 295, 296, 300, 334, 336, 276, 282, 283, 285],
        "right_eye": [160, 33, 161, 163, 133, 7, 173, 144, 145, 246, 153, 154, 155, 157, 158, 159],
        "right_eyebrow": [65, 66, 70, 105, 107, 46, 52, 53, 55, 63],
        "face_oval": [132, 389, 136, 10, 397, 400, 148, 149, 150, 21, 152, 284, 288, 162, 297,
                      172, 176, 54, 58, 323, 67, 454, 332, 338, 93, 356, 103, 361, 234, 109, 365,
                      379, 377, 378, 251, 127],
        "nose": [1, 2, 4, 5, 6, 19, 275, 278, 294, 168, 45, 48, 440,
                 64, 195, 197, 326, 327, 344, 220, 94, 97, 98, 115],
        "contours": [0, 7, 10, 13, 14, 17, 21, 33, 37, 39, 40, 46, 52, 53, 54, 55, 58, 61, 63, 65, 66,
                     67, 70, 78, 80, 81, 82, 84, 87, 88, 91, 93, 95, 103, 105, 107, 109, 127, 132, 133,
                     136, 144, 145, 146, 148, 149, 150, 152, 153, 154, 155, 157, 158, 159, 160, 161, 162,
                     163, 172, 173, 176, 178, 181, 185, 191, 234, 246, 249, 251, 263, 267, 269, 270, 276,
                     282, 283, 284, 285, 288, 291, 293, 295, 296, 297, 300, 308, 310, 311, 312, 314, 317,
                     318, 321, 323, 324, 332, 334, 336, 338, 356, 361, 362, 365, 373, 374, 375, 377, 378,
                     379, 380, 381, 382, 384, 385, 386, 387, 388, 389, 390, 397, 398, 400, 402, 405, 409,
                     415, 454, 466],
        "tesselation": list(range(468))
    }
    i = [i for i in face_mesh_index_dict]
    if index_group_name not in i:
        print(f"Face Mesh Index Groups: {i}")
        return None
    else:
        return face_mesh_index_dict[index_group_name]
