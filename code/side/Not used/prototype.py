import pandas as pd
import os
#train
path = "../csvfiles/train"
directory = os.fsencode(path)

df = pd.DataFrame()

vname = "LukketSalto5_flipped"

for file in os.listdir(directory):
     filename = os.fsdecode(file)
     # a = pd.read_csv(path + "/" + filename)
     a = pd.read_pickle(path + "/" + filename)
     df = pd.concat([df, a])



#test
test = pd.read_pickle("../csvfiles/test/" + vname + ".pkl")
y_test = test["labels"]
X_test = test.drop(columns = ["labels"])


from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

y_train = df["labels"]
X_train = df.drop(columns = ["labels"])

# clf = LogisticRegression(random_state=0).fit(X_train, y_train)
clf = MLPClassifier(random_state=1, max_iter=300).fit(X_train, y_train)
preds = clf.predict(X_test)

print("Accuracy is :")
print(clf.score(X_test, y_test))

# print(preds)


#at = annotation()
import cv2
import cv2
import time
import PoseModule as pd
from annotation import annotation

videoName = vname
cap = cv2.VideoCapture("../VideoData/" + videoName + ".mp4")

# cap = cv2.VideoCapture(0)
pTime = 0
detector = pd.poseDetector()
labels = []
a = "beginning"
skeleton = []
condition = True
counter = 0

while condition:

    succes, img = cap.read()

    if not succes:
        print("Video Ended")
        break
    h, w, c = img.shape
    img = detector.findPose(img, draw=True)
    lmlist = detector.findPosition(img, draw=False)

    # if len(lmlist) != 0:
    #     detector.findAngle(img,12, 14, 16)

    if len(lmlist) == 0:
        continue

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(img, str(int(fps)), (70, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)

    key = cv2.waitKey(10)


    a = "Prediction : " + str(preds[counter])
    b = "Annotation : " + str(y_test[counter])


    cv2.putText(img, a, (100, 150), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 210), 2)
    cv2.putText(img, b, (100, 100), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 255), 2)
    cv2.imshow("Image", img)



    counter += 1

    labels.append(a)
    skeleton.append(lmlist)


