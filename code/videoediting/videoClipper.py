#awesomeproject.py
#this is our main script that takes an video as input, apply the HPE and save it as a csv file.
#by GMS
import cv2
import time

#annotater

name = "Video19"
path = "../../../OneDrive - ITU/Bachelor/raw_video/" + name + "_raw.mp4"

cap = cv2.VideoCapture(path)

# cap = cv2.VideoCapture(
pTime = 0

a = "beginning"

condition = True
img_array = {}
counter = 0
img_array[counter] = []
while condition:
    succes, img = cap.read()

    if not succes:
        break

    h, w, c = img.shape

    size = (w, h)
    # img = detector.findPose(img, draw=True)
    # lmlist = detector.findPosition(img, draw=False)

    # if len(lmlist) != 0:
    #     detector.findAngle(img,12, 14, 16)

    # if len(lmlist) == 0:
    #     continue

    cTime = time.time()
    # fps = 1 / (cTime - pTime)
    pTime = cTime

    # cv2.putText(img, str(int(fps)), (70, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)

    key = cv2.waitKey(40)

    if key & 0XFF == ord('c'):
        counter += 1
        img_array[counter] = []

    img_array[counter].append(img)



    # cv2.putText(img, a, (100, 100), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 255), 2)
    cv2.imshow("Image", img)


print(len(img_array))

    # cv2.imshow("Image", blacky)

    # labels.append(a)
    # skeleton.append(lmlist)

outpath = "../../../OneDrive - ITU/Bachelor/clean_video/" + name + "/"
for j in range(len(img_array)):

    out = cv2.VideoWriter(outpath + "video" + str(j+1) + '.avi', cv2.VideoWriter_fourcc(*'DIVX'), 15, size)

    for i in range(len(img_array[j])):
        out.write(img_array[j][i])
    out.release()
