import json
import pandas as pd
import math

class json_to_csv:

    def __init__(self, path_to_json):
        self.path_to_json = path_to_json


    def pad_video(self, length, video):

        skeleton = video["0"]
        labels = video["1"]

        rounds = math.ceil(length/len(skeleton))


        padded_video =  []
        skeleton_temp = skeleton
        labels_temp = labels

        for i in range(rounds):
            skeleton_temp.extend(skeleton)
            labels_temp.extend(labels)

        padded_video.append(skeleton_temp[:length])
        padded_video.append(labels_temp[:length])

        return padded_video

    def read_json(self):

        skeleton_json = open(self.path_to_json)
        full_skeleton = json.load(skeleton_json)

        return full_skeleton

    def make_df(self):


        print("making json")


        df = pd.DataFrame()

        full_skeleton = self.read_json()
        full_skeleton_padded = {}

        for key in full_skeleton.keys():

            full_skeleton_padded[key] = self.pad_video(300,full_skeleton[key])
            # print(key)

            for j in range(len(full_skeleton_padded[key][0])):

                data_point = full_skeleton_padded[key][0][j]

                label = full_skeleton_padded[key][1][j]


                x_values = []
                y_values = []

                for i in range(0,66,2):

                    x_values.append(data_point[i])

                for i in range(1,66,2):
                    y_values.append(data_point[i])
                d = {}
                # print(len(frame))

                for i in range(33):
                    d[str(i)+"x"] = [x_values[i]]
                    d[str(i) + "y"] = [y_values[i]]

                d['finals'] = label
                names = key.split('_')
                d['clipname'] = names[1]
                d['videoname'] = names[0]

                dframe = pd.DataFrame(data=d)
                df = pd.concat([df, dframe])


        return df


    # def df_to_csv(self, path):
    #
    #     df = self.make_df()
    #
    #     df.to_csv(path, index=False)



def main():


    path_to_kfold = r"C:\Users\sofu0\OneDrive - ITU\Bachelor\kfold"

    folders = ["fold0", "fold1", "fold2", "fold3", "fold4"]
    subfolders = ["sub_folder0", "sub_folder1", "sub_folder2", "sub_folder3"]

    for folder in folders:
        path_to_folder = path_to_kfold + "\\" + folder


        #create trainset
        path_to_train = path_to_folder + "\\tra.json"

        j_t_c = json_to_csv(path_to_train)
        data = j_t_c.make_df()
        data.to_csv(path_to_folder + "\\train_padded.csv", index=False)


        for subfolder in subfolders:

            path_to_subfolder = path_to_folder + "\\" + subfolder

            path_train = path_to_subfolder + "\\train.json"

            j_t_c = json_to_csv(path_train)
            data = j_t_c.make_df()
            data.to_csv(path_to_subfolder + "\\train_padded.csv", index=False)


if __name__ == '__main__':
    main()