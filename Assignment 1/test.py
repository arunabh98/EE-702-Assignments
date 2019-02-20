# -*- coding: utf-8 -*-
"""
Created on Tue Feb 19 18:48:49 2019

@author: aruna
"""
import cv2
ret, thresh = cv2.threshold(roi_radiance.astype(float), 0.5, 1, 0)
im2, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)