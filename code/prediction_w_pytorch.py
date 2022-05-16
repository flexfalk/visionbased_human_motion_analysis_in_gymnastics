#awesomeproject.py
#this is our main script that takes an video as input, apply the HPE and save it as a csv file.
#by GMS
from torch import nn, optim
import torch.nn.functional as F
import torch.utils.data as data
import torch
from torch.utils.data.sampler import SubsetRandomSampler
import pandas
import numpy as np
import cv2
import time
import PoseModule as pd
from CNN_file import CNN
from torch.autograd import Variable
from preprocessing import LivePreprocessor, PreProcesser
import pickle
from create_skeleton import create_skeleton


#import preprocessor
pp = PreProcesser()
live_preproc = LivePreprocessor()


#import model
path = r"C:\Users\sofu0\OneDrive - ITU\Bachelor\models\CNN\150_iter_92_accuracy"
model = torch.load(path)
# model = CNN()
# model.load_state_dict(torch.load(path))
# model.eval()
print(model.eval())

#import labels and video
prelabels = pandas.read_csv(r'C:\Users\sofu0\OneDrive - ITU\Bachelor\clean_video\Video2\video1\majorityVote_clean.csv')
cap = cv2.VideoCapture(r'C:\Users\sofu0\OneDrive - ITU\Bachelor\clean_video\Video2\video1\1.mp4')

# cap = cv2.VideoCapture(r"C:\Users\sofu0\PycharmProjects\BACHELOR-ITU-2022\VideoData\LukketSalto5.mp4")
# cap = cv2.VideoCapture(0)
pTime = 0
detector = pd.poseDetector()
labels = []

#make file to save video
recording = True

frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

size = (frame_width, frame_height)

result_path = r"C:\Users\sofu0\OneDrive - ITU\Bachelor\models\CNN\example_videos\5.avi"
result = cv2.VideoWriter(result_path,
                         cv2.VideoWriter_fourcc(*'MJPG'),
                         10, size)



nums_to_labels = {0: "idle", 1: "take-off", 2: "skill", 3: "landing"}
sk_pp = create_skeleton()

condition = True
start_frame = True


all_frames = 0
n_succes = 0
counter = 0
tmp_x_var = 0
while condition:

    succes, img = cap.read()
    preds = "NaN"

    if not succes:
        break
    h, w, c = img.shape
    img = detector.findPose(img, draw=True)
    _, lmlist = detector.findPosition2(img, draw=False)
    # lmlist = detector.findPosition(img, draw=False)

    if len(lmlist) == 0:
        # print("noskelly")
        counter += 1
        continue

    if len(lmlist) != 0:


        all_frames += 1

        #make lmlist to pandas dataframe
        skelly = sk_pp.create_pandas_frame([lmlist])

        #finding start_frame. Important for normalizing.
        if start_frame:
            if tmp_x_var == 0:
                s_frame = skelly.copy()
                tmp_x_var += 1

            start_frame=False


        s_frame2 = s_frame.copy()

        skelly = pp.normalize_live(skelly, s_frame2)

        data = torch.Tensor(skelly.values.reshape(1, 1, 2, 33).astype(float))


        #predicts
        scores = model(data)
        _, preds = scores.max(1)
        preds = nums_to_labels[preds.numpy()[0]]



    label = prelabels["final"].to_list()[counter]

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    key = cv2.waitKey(100)

    cv2.putText(img, str("labels " +label), (100, 100), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 255), 2)
    similar = "false"
    color = (124, 0, 255)
    if label == preds:
        similar = "correct"
        color = (124, 252, 0)
        n_succes += 1
    cv2.putText(img, str("predictions " + str(preds)), (100, 150), cv2.FONT_HERSHEY_PLAIN, 2, color,2)
    # cv2.putText(img, similar, (100, 300), cv2.FONT_HERSHEY_PLAIN, 2, color, 2)
    # imgs = cv2.resize(img, (960, 540))
    cv2.imshow("Image", img)
    counter += 1
    result.write(img)

if recording:
    cap.release()
    result.release()


print(all_frames)
print(n_succes)
print("Accuracy: ", n_succes/all_frames)

