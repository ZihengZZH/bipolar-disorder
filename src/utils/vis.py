import os
import cv2
import json
import numpy as np
import pandas as pd

from smart_open import smart_open
from src.utils.io import get_sample


'''
ALL PRE-PROCESS FUNCTIONS
------------------------------------
visualize_landmarks(partition, index=None, verbose=False)
    visualize facial landmarks in interview videos
'''


# load the external configuration file
data_config = json.load(open('./config/data.json', 'r'))


def visualize_landmarks(partition, index=None, no_frame=False):
    """visualize facial landmarks in interview videos
    -----
    # para partition: train/dev/test set
    # para index: index of train/dev/test set
    """
    if partition not in ['train', 'dev', 'test']:
        print("\nerror input argument")
        return 
    index = 1 if not index else index

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
    poses_T = ['pose_Tx','pose_Ty','pose_Tz']
    poses_R = ['pose_Rx','pose_Ry','pose_Rz']

    if not video.isOpened():
        print("\nerror opening video file %s" % filename)

    while video.isOpened():
        _, frame = video.read()
        if no_frame:
            frame = np.zeros(shape=frame.shape)
        focal_length = frame.shape[1]
        camera_centre = (frame.shape[1] / 2, frame.shape[0] / 2)
        camera_matrix = np.array(
            [[focal_length, 0, camera_centre[0]],
            [0, focal_length, camera_centre[1]],
            [0, 0, 1]], dtype='double'
        )
        dist_coeffs = np.zeros((4,1))
        face_size = 50

        time = video.get(cv2.CAP_PROP_POS_MSEC)
        index = video.get(cv2.CAP_PROP_POS_FRAMES)
        print("frames: %d --- times: %f" % (index, time/1000))
        landmarks_match = landmarks[landmarks['timestamp'] == time/1000]

        if landmarks_match.index.any():
            for i in range(68):
                x, y = landmarks_match[coordinates[i]], landmarks_match[coordinates[68 + i]]
                x, y = int(float(x)), int(float(y))
                cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)

            vector_T = np.array(landmarks_match[poses_T])
            vector_R = np.array(landmarks_match[poses_R])

            point_3d = []
            rear_size = face_size
            rear_depth = 0
            point_3d.append((-rear_size, -rear_size, rear_depth))
            point_3d.append((-rear_size, rear_size, rear_depth))
            point_3d.append((rear_size, rear_size, rear_depth))
            point_3d.append((rear_size, -rear_size, rear_depth))
            point_3d.append((-rear_size, -rear_size, rear_depth))

            front_size = rear_size + 50
            front_depth = -100
            point_3d.append((-front_size, -front_size, front_depth))
            point_3d.append((-front_size, front_size, front_depth))
            point_3d.append((front_size, front_size, front_depth))
            point_3d.append((front_size, -front_size, front_depth))
            point_3d.append((-front_size, -front_size, front_depth))

            point_3d = np.array(point_3d, dtype=np.float).reshape(-1, 3)

            # Map to 2d image points
            (point_2d, _) = cv2.projectPoints(point_3d,
                                            vector_R,
                                            vector_T,
                                            camera_matrix,
                                            dist_coeffs)
            point_2d = np.int32(point_2d.reshape(-1, 2))

            # Draw all the lines
            color = (255,255, 0)
            line_width = 2
            cv2.polylines(frame, [point_2d], True, color, line_width, cv2.LINE_AA)
            cv2.line(frame, tuple(point_2d[1]), tuple(
                point_2d[6]), color, line_width, cv2.LINE_AA)
            cv2.line(frame, tuple(point_2d[2]), tuple(
                point_2d[7]), color, line_width, cv2.LINE_AA)
            cv2.line(frame, tuple(point_2d[3]), tuple(
                point_2d[8]), color, line_width, cv2.LINE_AA)

            if index == 45 or index == 285:
                cv2.imwrite(os.path.join('images', 'facial', 'frame_%d_face_%d.png' % (index, no_frame)), frame)
            
            if not no_frame:
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

def visualize_reconstrcution(no_frame=True):
    model_path_AV = smart_open('./pre-trained/DDAE/model_list.txt', 'rb', encoding='utf-8')
    model_list_AV = []

    for _, line_AV in enumerate(model_path_AV):
        line_AV = str(line_AV).replace('\n', '')
        model_list_AV.append(line_AV[:-2])

    landmarks_recon = np.load(os.path.join(model_list_AV[5], 'recon_1_train01.npy'))
    pose_recon = np.load(os.path.join(model_list_AV[5], 'recon_3_train01.npy'))

    video_dir = data_config['data_path_700']['recordings']
    landmark_dir = data_config['baseline_preproc']['AU_landmarks']
    if not os.path.isdir(video_dir):
        print("\nerror without recordings available")
        print("\nplease inject external hard drive")
        return
    
    filename = get_sample('train', 1)
    video = cv2.VideoCapture(os.path.join(video_dir, filename+'.mp4'))
    landmarks = pd.read_csv(os.path.join(landmark_dir, filename+'.csv'))
    coordinates = ['%s_%d' % (xy, i) for xy in ['x', 'y'] for i in range(68)]

    if not video.isOpened():
        print("\nerror opening video file %s" % filename)
    
    while video.isOpened():
        _, frame = video.read()
        if no_frame:
            frame = np.zeros(shape=frame.shape)
        focal_length = frame.shape[1]
        camera_centre = (frame.shape[1] / 2, frame.shape[0] / 2)
        camera_matrix = np.array(
            [[focal_length, 0, camera_centre[0]],
            [0, focal_length, camera_centre[1]],
            [0, 0, 1]], dtype='double'
        )
        dist_coeffs = np.zeros((4,1))
        face_size = 50

        time = video.get(cv2.CAP_PROP_POS_MSEC)
        index = video.get(cv2.CAP_PROP_POS_FRAMES)
        print("frames: %d --- times: %f" % (index, time/1000))
        landmarks_match = landmarks_recon[landmarks['timestamp'] == time/1000]
        pose_match = pose_recon[landmarks['timestamp'] == time/1000]

        if len(landmarks_match):
            for i in range(68):
                x, y = landmarks_match[0][i], landmarks_match[0][68+i]
                x, y = int(float(x)), int(float(y))
                cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)

            vector_T = pose_match[0][:3]
            vector_R = pose_match[0][3:]

            point_3d = []
            rear_size = face_size
            rear_depth = 0
            point_3d.append((-rear_size, -rear_size, rear_depth))
            point_3d.append((-rear_size, rear_size, rear_depth))
            point_3d.append((rear_size, rear_size, rear_depth))
            point_3d.append((rear_size, -rear_size, rear_depth))
            point_3d.append((-rear_size, -rear_size, rear_depth))

            front_size = rear_size + 50
            front_depth = -100
            point_3d.append((-front_size, -front_size, front_depth))
            point_3d.append((-front_size, front_size, front_depth))
            point_3d.append((front_size, front_size, front_depth))
            point_3d.append((front_size, -front_size, front_depth))
            point_3d.append((-front_size, -front_size, front_depth))

            point_3d = np.array(point_3d, dtype=np.float).reshape(-1, 3)

            # Map to 2d image points
            (point_2d, _) = cv2.projectPoints(point_3d,
                                            vector_R,
                                            vector_T,
                                            camera_matrix,
                                            dist_coeffs)
            point_2d = np.int32(point_2d.reshape(-1, 2))

            # Draw all the lines
            color = (255,255, 0)
            line_width = 2
            cv2.polylines(frame, [point_2d], True, color, line_width, cv2.LINE_AA)
            cv2.line(frame, tuple(point_2d[1]), tuple(
                point_2d[6]), color, line_width, cv2.LINE_AA)
            cv2.line(frame, tuple(point_2d[2]), tuple(
                point_2d[7]), color, line_width, cv2.LINE_AA)
            cv2.line(frame, tuple(point_2d[3]), tuple(
                point_2d[8]), color, line_width, cv2.LINE_AA)

            if index == 45 or index == 285:
                cv2.imwrite(os.path.join('images', 'facial', 'recons_%d_face_%d_multiDDAE_egemaps.png' % (index, no_frame)), frame)
            
            if not no_frame:
                cv2.putText(frame, 'Press q to quit', 
                            (10, 50),
                            cv2.FONT_HERSHEY_DUPLEX,
                            1,
                            (255, 255, 255),
                            1)
            cv2.imshow("facial landmarks on partition %s" % 'train', frame)
            key = cv2.waitKey(1) % 0xFF

            if key == ord('q'):
                print(model_list_AV[5])
                break