"""
Date: 07.31.2024

Status: Public

This file can be used in all projects.
"""

import cv2
import numpy as np
from Utilities.Common_Utils import EuclideanDistance

def LineSegment(img,pt_1,pt_2,color,thickness=1):
    radius = int(2.5 * thickness)
    cv2.line(img,pt_1,pt_2,color,thickness)
    cv2.circle(img,pt_1,radius,color,-1)
    cv2.circle(img,pt_2,radius,color,-1)

def Ray(img,pt_1,pt_2,color,thickness=1):
    radius = int(2.5 * thickness)
    angle = np.arctan2(pt_2[1] - pt_1[1],pt_2[0] - pt_1[0])
    head_length = 4 * thickness
    head_angle = np.pi / 6
    x_1 = int(pt_2[0] - head_length * np.cos(angle - head_angle))
    y_1 = int(pt_2[1] - head_length * np.sin(angle - head_angle))
    x_2 = int(pt_2[0] - head_length * np.cos(angle + head_angle))
    y_2 = int(pt_2[1] - head_length * np.sin(angle + head_angle))
    cv2.line(img,pt_1,pt_2,color,thickness)
    cv2.line(img,pt_2,(x_1,y_1),color,thickness)
    cv2.line(img,pt_2,(x_2,y_2),color,thickness)
    cv2.circle(img,pt_1,radius,color,-1)

def CornerRectangle(img,pt1,pt2,color,length=0,thickness=1):
    x1,y1 = pt1
    x2,y2 = pt2
    if length <= 0:
        length = int(min(EuclideanDistance((x2,y1),(x1,y1)),EuclideanDistance((x1,y1),(x1,y2))) * 0.25)
    cv2.line(img,(x1,y1),(x1 + length,y1),color,thickness)
    cv2.line(img, (x1, y1), (x1, y1 + length), color, thickness)
    cv2.line(img, (x1, y2), (x1, y2 - length), color, thickness)
    cv2.line(img, (x1, y2), (x1 + length, y2), color, thickness)
    cv2.line(img, (x2, y1), (x2 - length, y1), color, thickness)
    cv2.line(img, (x2, y1), (x2, y1 + length), color, thickness)
    cv2.line(img, (x2, y2), (x2, y2 - length), color, thickness)
    cv2.line(img, (x2, y2), (x2 - length, y2), color, thickness)

def HorizontalSquareBrackets(img,pt1,pt2,color,length=0,thickness=1):
    x1, y1 = pt1
    x2, y2 = pt2
    if length <= 0:
        length = int(min(EuclideanDistance((x2, y1), (x1, y1)), EuclideanDistance((x1, y1), (x1, y2))) * 0.25)
    cv2.line(img, (x1, y1), (x2,y1), color, thickness)
    cv2.line(img, (x1, y2), (x2, y2), color, thickness)
    cv2.line(img, (x1,y1), (x1, y1 + length), color, thickness)
    cv2.line(img, (x1, y2), (x1, y2 - length), color, thickness)
    cv2.line(img, (x2, y1), (x2, y1 + length), color, thickness)
    cv2.line(img, (x2, y2), (x2, y2 - length), color, thickness)

def SquareBrackets(img,pt1,pt2,color,length=0,thickness=1):
    x1, y1 = pt1
    x2, y2 = pt2
    if length <= 0:
        length = int(min(EuclideanDistance((x2, y1), (x1, y1)), EuclideanDistance((x1, y1), (x1, y2))) * 0.25)
    cv2.line(img, (x1, y1), (x1, y2), color, thickness)
    cv2.line(img, (x2, y1), (x2, y2), color, thickness)
    cv2.line(img, (x1, y1), (x1 + length, y1), color, thickness)
    cv2.line(img, (x1, y2), (x1 + length, y2), color, thickness)
    cv2.line(img, (x2, y1), (x2 - length, y1), color, thickness)
    cv2.line(img, (x2, y2), (x2 - length, y2), color, thickness)
