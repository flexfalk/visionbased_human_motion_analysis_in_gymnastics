import pandas as pd
import cv2
import os

class annotator:

    def __init__(self):
        pass

    def annotate_video(self, path_to_video) -> pd.DataFrame:

        # videoName = "LukketSalto4"

        # cap = cv2.VideoCapture(r"C:\Users\sofu0\OneDrive - ITU\Bachelor\clean_video\Video2\video2\2.mp4")
        cap = cv2.VideoCapture(path_to_video)
        pTime = 0
        labels = []
        a = None

        while True:
            succes, img = cap.read()
            key = cv2.waitKey(0)
            cv2.imshow("Image", img)

            #
            if key & 0XFF == ord('i'):
                a = "idle"
                break

            elif key & 0XFF == ord('t'):
                a = "take-off"
                break

            elif key & 0XFF == ord('l'):
                a = "landing"
                break

            elif key & 0XFF == ord('s'):
                a = "skill"
                break
            #
        cv2.destroyAllWindows()



        cap = cv2.VideoCapture(path_to_video)
        condition = True
        while condition:
            succes, img = cap.read()

            if not succes:
                break

            # cTime = time.time()
            # fps = 1 / (cTime - pTime)
            # pTime = cTime


            #
            key = cv2.waitKey(0)

            if key & 0XFF == ord('i'):
                a = "idle"

            elif key & 0XFF == ord('t'):
                a = "take-off"

            elif key & 0XFF == ord('l'):
                a = "landing"

            elif key & 0XFF == ord('s'):
                a = "skill"

            # cv2.putText(img, str(int(fps)), (70, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)
            cv2.putText(img, a, (100, 100), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 255), 2)
            cv2.imshow("Image", img)


            labels.append(a)


        df = pd.DataFrame(labels)
        df = df.rename(columns={0: "labels"})

        cv2.destroyAllWindows()

        return df


    def annotate_folder(self, path_to_folder, name):

        counter = len(os.listdir(path_to_folder))
        i = 0
        while i < counter:

            print(i)

            subfolder = os.listdir(path_to_folder)[i]
            path_to_sub_sub_folder = path_to_folder + r'\\' + subfolder

            for file in os.listdir(path_to_sub_sub_folder):
                if file[-4:] == '.mp4':
                    videoName = file
                else:
                    pass

            path_to_video = path_to_sub_sub_folder + r"\\" +  videoName

            print(path_to_video)

            df = self.annotate_video(path_to_video)
            csv_name = videoName[:-4] + "_" + name + ".csv"
            csv_path = path_to_sub_sub_folder + r"\\" + csv_name
            df.to_csv(csv_path)

            answer = input("want to redo?[y/n] \n")

            if answer == "n":
                i += 1
            if answer == "y":
                pass
        print('you are done :)')



def main():

    an = annotator()

    an.annotate_folder(r'C:\Users\sofu0\OneDrive - ITU\Bachelor\clean_video\Video19', "sofus")


if __name__ == "__main__":
    main()