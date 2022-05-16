import pandas as pd
import cv2
import os

class majortityVote:

    def __init__(self):
        pass


    def makeVote(self, file1, file2, file3):

        #read in the 3 csv files
        df1 = pd.read_csv(file1)
        df2 = pd.read_csv(file2)
        df3 = pd.read_csv(file3)

        #checking that they all have same lenght
        if (len(df1) != len(df2)) or (len(df2) != len(df1)):
            print("failure")
            return

        #merging them together
        q = df1.merge(df2, on='Unnamed: 0', how='left')
        q = q.merge(df3, on = 'Unnamed: 0', how='left')
        q = q.rename(columns={"labels":"labels_z"})

        #making 4'th column with final label. Start with being Unlabeled
        q["final"] = "Unlabeled"

        #go through each row and see if 2 aggre. either x & y, x & z or y & z. If agree setting the label to that
        for i in range(len(q)):

            if (q["labels_x"].iloc[i] == q["labels_y"].iloc[i]):
                q["final"].iloc[i] = q["labels_x"].iloc[i]

            elif (q["labels_x"].iloc[i] == q["labels_z"].iloc[i]):
                q["final"].iloc[i] = q["labels_x"].iloc[i]

            elif (q["labels_y"].iloc[i] == q["labels_z"].iloc[i]):
                q["final"].iloc[i] = q["labels_y"].iloc[i]

        return q



    def make_majortityVote(self, path_to_folder):

        #list the folders, find number of folders
        counter = len(os.listdir(path_to_folder))

        #go through eahch folder
        i = 0
        while i < counter:

            #list the subfolders inside the folder. Example Video1/video1, Video1/video2/ Video1/video3
            subfolder = os.listdir(path_to_folder)[i]
            #finding the path to the subfolder, by comining path of folder with name of subfolder
            path_to_sub_sub_folder = path_to_folder + r'\\' + subfolder

            #go through the content in the subfolder. Here there is 3 csv files with labels and 1 .mp4 file of the video
            csv_files = []
            for file in os.listdir(path_to_sub_sub_folder):
                #if its a csv file, add the path of that to the csv_files list
                if file[-4:] != '.mp4':
                    csvName = file
                    path_to_csv_file = path_to_sub_sub_folder + r'\\' + csvName
                    csv_files.append(path_to_csv_file)
                else:
                    pass

            #make majority vote of the 3 csv files
            final_df = self.makeVote(csv_files[0], csv_files[1], csv_files[2])

            #save path and export
            path_to_csv = path_to_sub_sub_folder + r'\\' + "majorityVote.csv"
            final_df.to_csv(path_to_csv)
            i += 1

def main():

    #enter path to your folder. Example ~\Bachelor\clean_video\Video1
    folderpath = r'C:\Users\sofu0\OneDrive - ITU\Bachelor\clean_video\Video11'

    #make majorityVote object, and call make_majorityVote
    mV = majortityVote()
    mV.make_majortityVote(folderpath)

if __name__ == "__main__":
    main()
