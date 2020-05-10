#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May  9 19:20:57 2020

@author: dg
"""

import cv2
import os
import numpy as np
import pandas as pd
from argparse import ArgumentParser
from PersonDetectionModel import PersonDetectionModel
from FaceDetectionModel import FaceDetectionModel
from AgeGenderRecognitionModel import AgeGenderRecognitionModel
from PersonReidentificationModel import PersonReidentificationModel

def build_argparser():
    """
    Parse command line arguments.

    :return: command line arguments
    """
    parser = ArgumentParser()

    parser.add_argument("-i", "--input", required=True, type=str,
                        help="Path to video file or enter cam for webcam")
    parser.add_argument("-l", "--cpu_extension", required=False, type=str,
                        default=None,
                        help="MKLDNN (CPU)-targeted custom layers."
                             "Absolute path to a shared library with the"
                             "kernels impl.")
    parser.add_argument("-d", "--device", type=str, required=False, default="CPU",
                        help="Specify the target device to infer on: "
                             "CPU, GPU, FPGA or MYRIAD is acceptable. Sample "
                             "will look for a suitable plugin for device "
                             "specified (CPU by default)")
    parser.add_argument("-pt", "--prob_threshold", type=float, required=False, default=0.6,
                        help="Probability threshold for detections filtering"
                        "(0.5 by default)")
    parser.add_argument("-is_export_csv", "--export_csv", type=bool, required=False, default=True,
                        help="If you want to export data as csv file." 
                        "Data contains image_id, age, gender")
    parser.add_argument("-image_dir", "--image_dir", type=str, required=False, default=None,
                        help="If you want to save pedestrian images then specify the directory.")
    return parser

def cos_similarity(X, Y):
    Y = Y.T    # (1, 256) x (256, n) = (1, n)
    return np.dot(X, Y) / (np.linalg.norm(X) * np.linalg.norm(Y, axis=0))

def identify_age_gender(person_img, frame, person_coords, img_id, prob_threshold, fd, agd):
    age=None
    gender=None
    cropped_face, face_coord = fd.predict(person_img, prob_threshold)
    if not len(face_coord)==0:
        age, gender = agd.predict(cropped_face)
        if gender==0:
            gender='F'
        else:
            gender='M'
        age = int(age)
        cv2.putText(frame, "age:"+str(int(age))+" gender:"+str(gender), (person_coords[0]-10,person_coords[1]+5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0),1)
        cv2.putText(frame, "id:"+str(img_id), (person_coords[0]-10,person_coords[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0),1)
    return frame, age, gender

def webcam_processing(frame, pdm, prm, fd, agd, prob_threshold, id_vec, person_id, image_dir):
    img_id=1
    count=0
    PRESENT = False
    persons_coords, count = pdm.predict(frame.copy(),prob_threshold)
    if not count==0:
        for person_coords in persons_coords:
            person_img = frame[person_coords[1]:person_coords[3],person_coords[0]:person_coords[2]]
            rei_vector = prm.predict(person_img)
            if len(id_vec)==0:
                frame, age, gender = identify_age_gender(person_img, frame, person_coords, img_id, prob_threshold, fd, agd)
                id_vec[person_id]=[rei_vector, age, gender]
                if not image_dir==None:
                    cv2.imwrite(image_dir+'/'+str(img_id)+'.jpg', person_img)
                person_id+=1
            else:
                for i in id_vec.keys():
                    res = cos_similarity(rei_vector, id_vec[i][0])
                    if res>0.45:
                        img_id=i
                        PRESENT=True
                        if id_vec[img_id][1]==None or id_vec[img_id][2]==None:
                            frame, age, gender = identify_age_gender(person_img, frame, person_coords, img_id, prob_threshold, fd, agd)
                            id_vec[img_id][1] = age
                            id_vec[img_id][2] = gender
                        break
                    PRESENT=False
        
                if not PRESENT:
                    img_id = person_id
                    person_id+=1
                    if not image_dir==None:
                        cv2.imwrite(image_dir+'/'+str(img_id)+'.jpg', person_img)
                    frame, age, gender = identify_age_gender(person_img, frame, person_coords, img_id, prob_threshold, fd, agd)
                    id_vec[img_id] = [rei_vector, age, gender]
                print(img_id, id_vec[img_id][1], id_vec[img_id][2])
            cv2.rectangle(frame, (person_coords[0],person_coords[1]), (person_coords[2],person_coords[3]), (255,0,0),2)
            
    cv2.putText(frame, "count:"+str(count), (40,40), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)
    return frame, id_vec, person_id

def video_processing(frame, pdm, prm, fd, agd, prob_threshold, id_vec, person_id, image_dir):
    frame_h = frame.shape[0]
    frame_w = frame.shape[1]
    img_id=1
    count=0
    PRESENT = False
    persons_coords, count = pdm.predict(frame.copy(),prob_threshold)
    if not count==0:
        for person_coords in persons_coords:
            x_cent = (person_coords[0] + person_coords[2])//2
            y_cent = (person_coords[1] + person_coords[3])//2
            if (y_cent<=frame_h//2-65) and (y_cent>=frame_h//2-95):
                person_img = frame[person_coords[1]:person_coords[3],person_coords[0]:person_coords[2]]
                rei_vector = prm.predict(person_img)
                if len(id_vec)==0:
                    frame, age, gender = identify_age_gender(person_img, frame, person_coords, img_id, prob_threshold, fd, agd)
                    id_vec[person_id]=[rei_vector, age, gender]
                    if not image_dir==None:
                        cv2.imwrite(image_dir+'/'+str(img_id)+'.jpg', person_img)
                    person_id+=1
                else:
                    for i in id_vec.keys():
                        res = cos_similarity(rei_vector, id_vec[i][0])
                        if res>0.45:
                            img_id=i
                            PRESENT=True
                            if id_vec[img_id][1]==None or id_vec[img_id][2]==None:
                                frame, age, gender = identify_age_gender(person_img, frame, person_coords, img_id, prob_threshold, fd, agd)
                                id_vec[img_id][1] = age
                                id_vec[img_id][2] = gender
                            break
                        PRESENT=False
            
                    if not PRESENT:
                        img_id = person_id
                        person_id+=1
                        if not image_dir==None:
                            cv2.imwrite(image_dir+'/'+str(img_id)+'.jpg', person_img)
                        frame, age, gender = identify_age_gender(person_img, frame, person_coords, img_id, prob_threshold, fd, agd)
                        id_vec[img_id] = [rei_vector, age, gender]
                print(img_id, id_vec[img_id][1], id_vec[img_id][2])
            cv2.rectangle(frame, (person_coords[0],person_coords[1]), (person_coords[2],person_coords[3]), (255,0,0),2)
            
    cv2.line(frame, (0, frame_h//2-80), (frame_w, frame_h//2-80), (0,0,255),2)
    cv2.putText(frame, "count:"+str(count), (40,40), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)
    return frame, id_vec, person_id

def main(args):
    deviceType = args.device
    cpuExt = args.cpu_extension
    probThresh = args.prob_threshold
    filePath = args.input
    is_export = args.export_csv
    image_dir = args.image_dir
    prob_threshold = probThresh
    
    pdm = PersonDetectionModel("intel/person-detection-retail-0013/FP32/person-detection-retail-0013.xml",deviceType,cpuExt)
    pdm.load_model()
    fd = FaceDetectionModel("intel/face-detection-retail-0005/FP32/face-detection-retail-0005.xml",deviceType,cpuExt)
    fd.load_model()
    prm = PersonReidentificationModel("intel/person-reidentification-retail-0300/FP32/person-reidentification-retail-0300.xml",deviceType,cpuExt)
    prm.load_model()
    agd = AgeGenderRecognitionModel("intel/age-gender-recognition-retail-0013/FP32/age-gender-recognition-retail-0013.xml",deviceType,cpuExt)
    agd.load_model()
    
    if filePath=='cam':
        camera = cv2.VideoCapture(0)
    else:
        camera = cv2.VideoCapture(filePath)
    
    id_vec = {}
    person_id=1
    while camera.isOpened():
        ret, frame = camera.read()
    
        if not ret:
            break
        key = cv2.waitKey(60)
        if filePath=='cam':
            frame, id_vec, person_id = webcam_processing(frame, pdm, prm, fd, agd, prob_threshold, id_vec, person_id, image_dir)
        else:
            frame, id_vec, person_id = video_processing(frame, pdm, prm, fd, agd, prob_threshold, id_vec, person_id, image_dir)
        
        cv2.imshow('video',cv2.resize(frame,(768,456)))
        if key==27:
            break
    if is_export:
        data = [[i, id_vec[i][1], id_vec[i][2]]for i in id_vec.keys()]
        df = pd.DataFrame(data = data, columns = ['image_id', 'age', 'gender'])
        df.to_csv('data.csv', index=False)
    cv2.destroyAllWindows()
    camera.release()
    
if __name__ == '__main__':
    args = build_argparser().parse_args()
    main(args)
    
    