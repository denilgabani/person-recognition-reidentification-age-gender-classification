#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May  9 19:20:57 2020

@author: dg
"""

import cv2
import numpy as np
import pandas as pd
from PersonDetectionModel import PersonDetectionModel
from FaceDetectionModel import FaceDetectionModel
from AgeGenderRecognitionModel import AgeGenderRecognitionModel
from PersonReidentificationModel import PersonReidentificationModel as Prim

def cos_similarity(X, Y):
    Y = Y.T    # (1, 256) x (256, n) = (1, n)
    return np.dot(X, Y) / (np.linalg.norm(X) * np.linalg.norm(Y, axis=0))

def age_gender_identify(person_img, fd, agd):
    age=None
    gender=None
    cropped_face, face_coord = fd.predict(person_img, 0.8)
    if not len(face_coord)==0:
        age, gender = agd.predict(cropped_face,0.6)
        if gender==0:
            gender='F'
        else:
            gender='M'
        age = int(age)
        cv2.putText(frame, "age:"+str(int(age))+" gender:"+str(gender), (person_coords[0]-10,person_coords[1]+5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0),1)
        cv2.putText(frame, "id:"+str(img_id), (person_coords[0]-10,person_coords[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0),1)
    return frame, age, gender

camera = cv2.VideoCapture("sample2.mp4")

pdm = PersonDetectionModel("intel/person-detection-retail-0013/FP32/person-detection-retail-0013.xml")
pdm.load_model()
fd = FaceDetectionModel("intel/face-detection-retail-0005/FP32/face-detection-retail-0005.xml")
fd.load_model()
prm = Prim("intel/person-reidentification-retail-0300/FP32/person-reidentification-retail-0300.xml")
prm.load_model()
agd = AgeGenderRecognitionModel("intel/age-gender-recognition-retail-0013/FP32/age-gender-recognition-retail-0013.xml")
agd.load_model()

count=0
id_vec = {}
res_list=[]
person_id=1
while camera.isOpened():
    ret, frame = camera.read()

    if not ret:
        break
    
    frame_h = frame.shape[0]
    frame_w = frame.shape[1]
    key = cv2.waitKey(60)
    img_id=1

    PRESENT = False
    persons_coords, count = pdm.predict(frame.copy(),0.6)
    if not count==0:
        for person_coords in persons_coords:
            x_cent = (person_coords[0] + person_coords[2])//2
            y_cent = (person_coords[1] + person_coords[3])//2
            if (y_cent<=frame_h//2-65) and (y_cent>=frame_h//2-95):
                person_img = frame[person_coords[1]:person_coords[3],person_coords[0]:person_coords[2]]
                rei_vector = prm.predict(person_img,0.6)
                if len(id_vec)==0:
                    frame, age, gender = age_gender_identify(person_img, fd, agd)
                    id_vec[person_id]=[rei_vector, age, gender]
                    cv2.imwrite('images/'+str(img_id)+'.jpg', person_img)
                    person_id+=1
                else:
                    for i in id_vec.keys():
                        res = cos_similarity(rei_vector, id_vec[i][0])
                        if res>0.45:
                            img_id=i
                            PRESENT=True
                            if id_vec[img_id][1]==None or id_vec[img_id][2]==None:
                                frame, age, gender = age_gender_identify(person_img, fd, agd)
                                id_vec[img_id][1] = age
                                id_vec[img_id][2] = gender
                            break
                        PRESENT=False
            
                    if not PRESENT:
                        img_id = person_id
                        person_id+=1
                        cv2.imwrite('images/'+str(img_id)+'.jpg', person_img)
                        frame, age, gender = age_gender_identify(person_img, fd, agd)
                        id_vec[img_id] = [rei_vector, age, gender]
                print(img_id, age, gender)
            cv2.rectangle(frame, (person_coords[0],person_coords[1]), (person_coords[2],person_coords[3]), (255,0,0),2)
            
    cv2.line(frame, (0, frame_h//2-80), (frame_w, frame_h//2-80), (0,0,255),2)
    cv2.putText(frame, "count:"+str(count), (40,40), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)
    cv2.imshow('video',cv2.resize(frame,(768,456)))
    if key==27:
        break
data = [[i, id_vec[i][1], id_vec[i][2]]for i in id_vec.keys()]
df = pd.DataFrame(data = data, columns = ['image_id', 'age', 'gender'])
df.to_csv('data.csv', index=False)
cv2.destroyAllWindows()
camera.release()



