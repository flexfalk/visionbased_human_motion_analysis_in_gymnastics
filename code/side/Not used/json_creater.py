import pandas as pd
import json
import os


class json_creater:

    def __init__(self):
        self.dict = {}

    def read_Xcsv(self, csv_path) -> pd.DataFrame:
        df = pd.read_csv(csv_path)
        df = df.drop(['Unnamed: 0'], axis=1)
        print(df.shape)
        lt = df.values.tolist()
        return lt

    def read_Ycsv(self, csv_path) -> pd.DataFrame:
        df = pd.read_csv(csv_path)
        df = df['final']
        lt = df.values.tolist()
        return lt

    def remove_skeleton(self, x_path, y_path):
        x = self.Xcsv(x_path)
        y = self.read_Ycsv(y_path)


    def combine(self, x_path, y_path, video_name) -> None:
        x = self.read_Xcsv(x_path)
        y = self.read_Ycsv(y_path)
        self.dict[video_name] = (x, y)

    def annotate_folder(self, path_to_folder):

        counter = len(os.listdir(path_to_folder))
        i = 0
        if path_to_folder[-7:-2] == 'Video' or path_to_folder[-7:-2] == 'video':
            folder_name = path_to_folder[-7:]
        else:
            folder_name = path_to_folder[-6:]

        while i < counter:

            subfolder = os.listdir(path_to_folder)[i]
            path_to_sub_sub_folder = path_to_folder + r'/' + subfolder

            if subfolder[0] != '.':
                for file in os.listdir(path_to_sub_sub_folder):
                    if file[-9:] == 'clean.csv':
                        Ycsv_name = file
                    if file == 'preprocessed_skeleton.csv':
                        Xcsv_name = file
                    if file[-4:] == '.mp4':
                        if len(file) == 7:
                            video_name = file[:-5]
                        else:
                            video_name = file[:-4]
                    else:
                        pass

                path_to_Xcsv = path_to_sub_sub_folder + r"/" + Xcsv_name
                path_to_Ycsv = path_to_sub_sub_folder + r"/" + Ycsv_name

                i += 1
                full_video_name = folder_name + '_' + video_name
                self.combine(path_to_Xcsv, path_to_Ycsv, full_video_name)

            else:
                i += 1

    def create_json(self, path_to_folder) -> None:
        self.annotate_folder(path_to_folder)
        path_to_json = r'/Users/Morten/Library/CloudStorage/OneDrive-SharedLibraries-ITU/Sofus Sebastian Schou Konglevoll - Bachelor/all_skeleton/skeleton_good_videos.json'
        with open(path_to_json, 'w') as outfile:
            json.dump(self.dict, outfile)


def main():
    good_videos = ["Video1", "Video2", "Video3", "Video5", "Video11", "Video13", "Video17", "Video19"]
    js_creater = json_creater()
    good_videos = [1,2,3,5,11,13,17,19]

    path = r'/Users/Morten/Library/CloudStorage/OneDrive-SharedLibraries-ITU/Sofus Sebastian Schou Konglevoll - Bachelor/clean_video'
    for i in good_videos:
        path_to_folder =  path + '/Video' + str(i)
        js_creater.create_json(path_to_folder)


if __name__ == '__main__':
    main()


