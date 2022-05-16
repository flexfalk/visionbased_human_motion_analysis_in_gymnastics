#PoseModule.py
#this is a module that makes it easy to interact with the cv2 and media pipe library. The class poseDetector is used to apply HPE in the awesomeproject.py
#By GMS

#1. test in PyCharm

import mediapipe as mp
import cv2
import time
import numpy as np
import pandas as pd


class poseDetector():

    def __init__(self, mode=False, upBody = False, smooth = True,
                 detectionCon = True, trackCon = True):

        self.mode = mode
        self.upBody = upBody
        self.smooth = smooth
        self.detectionCon = detectionCon
        self.trackCon = trackCon


        self.mpDraw = mp.solutions.drawing_utils
        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose(self.mode, self.upBody, self.smooth, self.detectionCon, self.trackCon)

    def findPose(self, img, draw=True):

        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(imgRGB)


        if self.results.pose_landmarks:
            if draw:
                self.mpDraw.draw_landmarks(img, self.results.pose_landmarks,
                                           self.mpPose.POSE_CONNECTIONS)

        return img

    def findPosition(self, img, draw=True):

        self.lmlist = []

        if self.results.pose_landmarks:
            for id, lm in enumerate(self.results.pose_landmarks.landmark):
                # print(self.results.pose_landmarks.landmark)
                h, w, c = img.shape

#               #print(id,lm)

                # cx, cy = int(lm.x * w), int(lm.y*h)

                #Not Scaledd with picture size
                cx, cy = float(lm.x), float(lm.y)
                self.lmlist.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 5, (255,0,0), cv2.FILLED)

        return self.lmlist

    def findPosition2(self, img, draw=True):

        self.lmlist = []
        self.for_show = []

        if self.results.pose_landmarks:
            for id, lm in enumerate(self.results.pose_landmarks.landmark):
                # print(self.results.pose_landmarks.landmark)
                h, w, c = img.shape

                # Not Scaledd with picture size
                cx, cy = float(lm.x * w), float(lm.y * h)
                cx_n, cy_n = float(lm.x), float(lm.y)
                self.for_show.append([id, cx, cy])
                self.lmlist.append([id, cx_n, cy_n])
                if draw:
                    cv2.circle(img, (cx, cy), 5, (255, 0, 0), cv2.FILLED)

        return self.lmlist, self.for_show

        #         if draw:
        #             cv2.circle(img, (cx, cy), 5, (255, 0, 0), cv2.FILLED)
        #
        # return self.lmlist

    def findAngle(self, img, p1, p2, p3, draw=True):

        _, x1, y1 = self.lmlist[p1]
        _, x2, y2 = self.lmlist[p2]
        _, x3, y3 = self.lmlist[p3]

        if draw:
            cv2.circle(img, (x1, y1), 10, (0, 0, 255), cv2.FILLED)
            cv2.circle(img, (x1, y1), 15, (0, 0, 255), 2)
            cv2.circle(img, (x2, y2), 10, (0, 0, 255), cv2.FILLED)
            cv2.circle(img, (x2, y2), 15, (0, 0, 255), 2)
            cv2.circle(img, (x3, y3), 10, (0, 0, 255), cv2.FILLED)
            cv2.circle(img, (x3, y3), 15, (0, 0, 255), 2)

def main():

    cap = cv2.VideoCapture("../VideoData/LukketSalto5.mp4")
    pTime = 0
    detector = poseDetector()
    while True:
        succes, img = cap.read()
        img = detector.findPose(img)

        lmlist = detector.findPosition(img, draw=False)

        if len(lmlist) != 0:
            # print(lmlist)

            cv2.circle(img, (lmlist[0][1], lmlist[0][2]), 15, (255, 0, 0), cv2.FILLED)


        # cTime = time.time()
        # fps = 1/ (cTime-pTime)
        # pTime = cTime
        # cv2.putText(img, str(int(fps)), (70,50), cv2.FONT_HERSHEY_PLAIN, 3, (255,0,0),3)
        #
        # cv2.imshow("Image", img)
        # cv2.waitKey(1)


if __name__ == "__main__":
    main()