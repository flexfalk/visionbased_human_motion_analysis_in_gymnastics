import mediapipe as mp
import cv2
import time
import numpy as np
import pandas as pd
import os

def main():


    mpPose = mp.solutions.pose
    pose = mpPose.Pose()
    mpDraw = mp.solutions.drawing_utils # For drawing keypoints
    points = mpPose.PoseLandmark # Landmarks
    path = r"C:\Users\Gustav Bakhauge\ITU\Sofus Sebastian Schou Konglevoll - Bachelor\clean_video\Video1\Video1\frames" # enter dataset path
    path_skele = r"C:\Users\Gustav Bakhauge\ITU\Sofus Sebastian Schou Konglevoll - Bachelor\clean_video\Video1\Video1\Video1_skeleton.csv" # enter dataset path
    skele_data = pd.read_csv(path_skele)
    skele_data = skele_data.drop("Unnamed: 0", axis=1)
    data = []

    for p in points:
        x = str(p)[13:]
        # print(x)
        data.append(x + "_x")
        data.append(x + "_y")
        # data.append(x + "_z")
        # data.append(x + "_vis")

    data = pd.DataFrame(columns = data) # Empty dataset
    count = 0
    print(data.shape)
    print(skele_data.shape)
    some_list = []

    for img in os.listdir(path):
        temp = []
        img = cv2.imread(path + "/" + img)

        imageWidth, imageHeight = img.shape[:2]
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # imgGrey = cv2.cvtColor(imgRGB, cv2.COLOR_BGR2GRAY)

        blackie = np.zeros(img.shape) # Blank image


        results = pose.process(imgRGB)
        vanillaresults = pose.process(img)
        # grey = pose.process(imgGrey)




        if results.pose_landmarks:
            # print(results.pose_landmarks) #xyz vis, values from 0:1

            mpDraw.draw_landmarks(imgRGB, results.pose_landmarks, mpPose.POSE_CONNECTIONS) #draw landmarks on image
            # mpDraw.draw_landmarks(img, results.pose_landmarks, mpPose.POSE_CONNECTIONS) # draw landmarks on blackie


        if vanillaresults.pose_landmarks:
            mpDraw.draw_landmarks(img, results.pose_landmarks, mpPose.POSE_CONNECTIONS)  # draw landmarks on blackie

            # landmarks = results.pose_landmarks.landmark
            # for i,j in zip(points,landmarks):
            #     temp = temp + [j.x, j.y, j.z, j.visibility]
            # data.loc[count] = temp
            # count +=1

        #
        imgS = cv2.resize(imgRGB, (1500, 800))
        # vanillS = cv2.resize(img, (960, 540))
        # blackieS = cv2.resize(blackie, (960, 540))
        cv2.imshow("Image", imgS)
        # cv2.imshow("Black",vanillS)
        cv2.waitKey(100)

if __name__=="__main__":
        main()

