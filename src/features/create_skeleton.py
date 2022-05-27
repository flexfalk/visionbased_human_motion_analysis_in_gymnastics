import cv2
import time
import PoseModule as pos
import pandas as pd
import os
# from preprocessing import PreProcesser

class create_skeleton:

    def __init__(self):
        pass

    def make_skeleton(self, path_to_video):

        # cap = cv2.VideoCapture(r'C:\Users\sofu0\OneDrive - ITU\Bachelor\clean_video\Video5\video2\video2.mp4')
        cap = cv2.VideoCapture(path_to_video)
        pTime = 0
        detector = pos.poseDetector()


        labels = []
        a = "beginning"
        skeleton = []
        condition = True

        while condition:
            succes, img = cap.read()

            if not succes:
                break


            # imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            # img = imgRGB # Make for different colors

            img = cv2.resize(img, (1500, 800))


            if not succes:
                break
            h, w, c = img.shape
            img = detector.findPose(img, draw=True)
            lmlist = detector.findPosition(img, draw=False)

            # if len(lmlist) != 0:
            #     detector.findAngle(img, 12, 14, 16)

            if len(lmlist) == 0:
                print("cant find")
                skeleton.append(lmlist)
                continue

            cTime = time.time()
            fps = 1 / (cTime - pTime)
            pTime = cTime
            cv2.putText(img, str(int(fps)), (70, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)

            # if key & 0XFF == ord('j'):
            #     a = "jump"
            # elif key & 0XFF == ord('l'):
            #     a = "landing"

            # cv2.putText(img, a, (100, 100), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 255), 2)

            cv2.imshow("Image", img)
            # # cv2.imshow("Image", blacky)
            cv2.waitKey(30)
            labels.append(a)
            skeleton.append(lmlist)
            print(skeleton)

        return skeleton

    def create_pandas_frame(self, skeleton):

        df = pd.DataFrame()

        for frame in skeleton:
            d = {}
            # print(len(frame))

            for i in range(33):
                if len(frame) == 0:
                    d[str(i) + "x"] = [0]
                    d[str(i) + "y"] = [0]
                    continue

                d[str(i)+"x"] = [frame[i][1]]
                d[str(i) + "y"] = [frame[i][2]]

            dframe = pd.DataFrame(data=d)
            df = pd.concat([df, dframe])

        return df


    def do_stuff_in_folder(self, path_to_folder):

        counter = len(os.listdir(path_to_folder))
        i = 0
        while i < counter:

            # print(i)

            subfolder = os.listdir(path_to_folder)[i]

            #For Windows
            # path_to_sub_sub_folder = path_to_folder + r'\\' + subfolder

            #For MAC
            path_to_sub_sub_folder = path_to_folder + r'/' + subfolder

            if subfolder[0] != '.':
                for file in os.listdir(path_to_sub_sub_folder):
                    if file[-4:] == '.mp4':
                        videoName = file
                    else:
                        pass

                #For Windows
                # path_to_video = path_to_sub_sub_folder + r"\\" + videoName

                #For MAC
                path_to_video = path_to_sub_sub_folder + r"/" + videoName

                # print(path_to_video)

                skeleton = self.make_skeleton(path_to_video)
                df = self.create_pandas_frame(skeleton)

                # print(df)
                # stop
                csv_name = videoName[:-4] + "init_skeleton-v2" + ".csv"
                #For Windows
                # csv_path = path_to_sub_sub_folder + r"\\" + csv_name

                #For Mac
                csv_path = path_to_sub_sub_folder + r"/" + csv_name
                # df["videoname"] = path_to_folder.split("\\")[-1]
                # pre = PreProcesser()
                # df = pre.normalize(df)


                # df.to_csv(csv_path)
                i += 1
                break
            else:
                i +=1

        print('you are done :)')


def main():

    cs = create_skeleton()
    # skeleton = cs.make_skeleton(path_to_video)
    # cs.create_pandas_frame(skeleton)


    for i in range(1, 20):
        # cs.do_stuff_in_folder(r'C:\Users\Gustav Bakhauge\ITU\Sofus Sebastian Schou Konglevoll - Bachelor\clean_video\Video' + str(i))
        # skeleton = cs.make_skeleton(r'C:\Users\Gustav Bakhauge\ITU\Sofus Sebastian Schou Konglevoll - Bachelor\clean_video\Video1\Video1\Video1.mp4')
        cs.do_stuff_in_folder(r'/Users/Morten/Library/CloudStorage/OneDrive-SharedLibraries-ITU/Sofus Sebastian Schou Konglevoll - Bachelor/clean_video/Video' + str(i))
        break






if __name__ == "__main__":
    main()





