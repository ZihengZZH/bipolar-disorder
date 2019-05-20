import os
import cv2
import json
import numpy as np
import pandas as pd

from src.utils.io import get_sample


'''
ALL PRE-PROCESS FUNCTIONS
------------------------------------
visualize_landmarks(partition, index=None, verbose=False)
    visualize facial landmarks in interview videos
'''


# load the external configuration file
data_config = json.load(open('./config/data.json', 'r'))


def visualize_landmarks(partition, index=None, verbose=False):
    """visualize facial landmarks in interview videos
    -----
    # para partition: train/dev/test set
    # para index: index of train/dev/test set
    """
    if partition not in ['train', 'dev', 'test']:
        print("\nerror input argument")
        return 
    index = 10 if not index else index

    video_dir = data_config['data_path_700']['recordings']
    landmark_dir = data_config['baseline_preproc']['AU_landmarks']
    if not os.path.isdir(video_dir):
        print("\nerror without recordings available")
        print("\nplease inject external hard drive")
        return

    filename = get_sample(partition, index)
    video = cv2.VideoCapture(os.path.join(video_dir, filename+'.mp4'))
    landmarks = pd.read_csv(os.path.join(landmark_dir, filename+'.csv'))
    coordinates = ['%s_%d' % (xy, i) for xy in ['x', 'y'] for i in range(68)]

    if not video.isOpened():
        print("\nerror opening video file %s" % filename)

    while video.isOpened():
        _, frame = video.read()
        # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        time = video.get(cv2.CAP_PROP_POS_MSEC)
        index = video.get(cv2.CAP_PROP_POS_FRAMES)
        print("frames: %d --- times: %f" % (index, time/1000))
        landmarks_match = landmarks[landmarks['timestamp'] == time/1000]

        if landmarks_match.index.any():
            for i in range(68):
                x, y = landmarks_match[coordinates[i]], landmarks_match[coordinates[68 + i]]
                x, y = int(float(x)), int(float(y))
                cv2.circle(frame, (x, y), 1, (255, 0, 0), -1)
            
            cv2.putText(frame, 'Press q to quit', 
                        (10, 50),
                        cv2.FONT_HERSHEY_DUPLEX,
                        1,
                        (255, 255, 255),
                        1)
            cv2.imshow("facial landmarks on partition %s" % partition, frame)
            key = cv2.waitKey(1) % 0xFF

            if key == ord('q'):
                break
