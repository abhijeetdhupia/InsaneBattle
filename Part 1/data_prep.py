"""
Build an orientation classifier which takes 17 keypoints as input and determines the orientation of the person. 
We have 7 classes for orientation:

(Class - 0): Standing and facing the camera: 
(Class - 1): Standing and facing left
(Class - 2): Standing and facing right
(Class - 3): Lying down on the back, facing left
(Class - 4): Lying down on the back, facing right
(class - 5): Lying in the pushup position, facing the camera
(class - 6): Lying in the pushup position, facing left
(class - 7): Lying in the pushup position, facing right

0 - Nose,
1 - Left Eye,
2 - Right Eye,
3 - Left Ear,
4 - Right Ear,
5 - Left Shoulder,
6 - Right Shoulder,
7 - Left Elbow,
8 - Right Elbow,
9 - Left Wrist,
10 - Right Wrist,
11 - Left Hip,
12 - Right Hip,
13 - Left Knee,
14 - Right Knee,
15 - Left Ankle,
16 - Right Ankle
"""

import numpy as np
import pandas as pd

# set seed
np.random.seed(42)


# Read the testing.json file 
test_df = pd.read_json('./data/testing.json')
test_df[[
    'nose_x','nose_y','nose_cs',
    'left_eye_x','left_eye_y','left_eye_cs',
    'right_eye_x','right_eye_y','right_eye_cs',
    'left_ear_x','left_ear_y','left_ear_cs',
    'right_ear_x','right_ear_y','right_ear_cs',
    'left_shoulder_x','left_shoulder_y','left_shoulder_cs',
    'right_shoulder_x','right_shoulder_y','right_shoulder_cs',
    'left_elbow_x','left_elbow_y','left_elbow_cs',
    'right_elbow_x','right_elbow_y','right_elbow_cs',
    'left_wrist_x','left_wrist_y','left_wrist_cs',
    'right_wrist_x','right_wrist_y','right_wrist_cs',
    'left_hip_x','left_hip_y','left_hip_cs',
    'right_hip_x','right_hip_y','right_hip_cs',
    'left_knee_x','left_knee_y','left_knee_cs',
    'right_knee_x','right_knee_y','right_knee_cs',
    'left_ankle_x','left_ankle_y','left_ankle_cs',
    'right_ankle_x','right_ankle_y','right_ankle_cs']] = pd.DataFrame(test_df.kps.tolist(), index= test_df.index)


testing_df = test_df[['id',
         'nose_x','nose_y',
         'left_eye_x','left_eye_y',
         'right_eye_x','right_eye_y',
         'left_ear_x','left_ear_y',
         'right_ear_x','right_ear_y',
         'left_shoulder_x','left_shoulder_y',
         'right_shoulder_x','right_shoulder_y',
         'left_elbow_x','left_elbow_y',
         'right_elbow_x','right_elbow_y',
         'left_wrist_x','left_wrist_y',
         'right_wrist_x','right_wrist_y',
         'left_hip_x','left_hip_y',
         'right_hip_x','right_hip_y',
         'left_knee_x','left_knee_y',
         'right_knee_x','right_knee_y',
         'left_ankle_x','left_ankle_y',
         'right_ankle_x','right_ankle_y']].round(4)

# save the test data 
testing_df.to_csv('./data/test_data.csv', index=False)

# Read the training.json file 
df = pd.read_json('./data/training.json')

# Divide an array column into multiple columns
df[[
    'nose_x','nose_y','nose_cs',
    'left_eye_x','left_eye_y','left_eye_cs',
    'right_eye_x','right_eye_y','right_eye_cs',
    'left_ear_x','left_ear_y','left_ear_cs',
    'right_ear_x','right_ear_y','right_ear_cs',
    'left_shoulder_x','left_shoulder_y','left_shoulder_cs',
    'right_shoulder_x','right_shoulder_y','right_shoulder_cs',
    'left_elbow_x','left_elbow_y','left_elbow_cs',
    'right_elbow_x','right_elbow_y','right_elbow_cs',
    'left_wrist_x','left_wrist_y','left_wrist_cs',
    'right_wrist_x','right_wrist_y','right_wrist_cs',
    'left_hip_x','left_hip_y','left_hip_cs',
    'right_hip_x','right_hip_y','right_hip_cs',
    'left_knee_x','left_knee_y','left_knee_cs',
    'right_knee_x','right_knee_y','right_knee_cs',
    'left_ankle_x','left_ankle_y','left_ankle_cs',
    'right_ankle_x','right_ankle_y','right_ankle_cs']] = pd.DataFrame(df.kps.tolist(), index= df.index)

# create a new dataframe with only the columns that we want
train_df = df[['id',
         'nose_x','nose_y',
         'left_eye_x','left_eye_y',
         'right_eye_x','right_eye_y',
         'left_ear_x','left_ear_y',
         'right_ear_x','right_ear_y',
         'left_shoulder_x','left_shoulder_y',
         'right_shoulder_x','right_shoulder_y',
         'left_elbow_x','left_elbow_y',
         'right_elbow_x','right_elbow_y',
         'left_wrist_x','left_wrist_y',
         'right_wrist_x','right_wrist_y',
         'left_hip_x','left_hip_y',
         'right_hip_x','right_hip_y',
         'left_knee_x','left_knee_y',
         'right_knee_x','right_knee_y',
         'left_ankle_x','left_ankle_y',
         'right_ankle_x','right_ankle_y']].round(4)

train_labels = df[['id','gt']]

# save train_df and train_labels to csv files
train_df.to_csv('./data/train_data.csv', index=False)
train_labels.to_csv('./data/train_labels.csv', index=False)