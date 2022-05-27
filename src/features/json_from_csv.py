import pandas as pd
import json


class Json_from_csv:

    def __init__(self):
        self.final_dict = {}
        self.file = None

    def read_csv(self, path):
        file_ = pd.read_csv(path)
        # file_ = file_.drop(['Unnamed: 0'], axis=1)
        # file_ = file_.drop(['Unnamed: 0.1'], axis=1)
        self.file = file_


    def make_dict(self):
        video_name = None
        clip_name = None


        for current_frame in self.file.iloc():
            frame = list(current_frame)

            if frame[-1] == video_name:
                if frame[-2] != clip_name:
                    labels = []
                    data_points = []
                    clip_name = frame[-2]
            else:
                clip_name = frame[-2]
                video_name = frame[-1]

            label = frame[-3]
            data_point = list(map(float, frame[:-3]))

            full_name = str(video_name) + '_' + str(clip_name)

            if full_name in self.final_dict.keys():
                self.final_dict[full_name][0].append(data_point)
                self.final_dict[full_name][1].append(label)
            else:
                self.final_dict[full_name] = [[data_point], [label]]

    def dict_to_json(self, path_to_csv):

        self.read_csv(path_to_csv)



        self.make_dict()

        path_to_json = r'/Users/Morten/Library/CloudStorage/OneDrive-SharedLibraries-ITU/Sofus Sebastian Schou Konglevoll - Bachelor/all_skeleton/skeleton_from_csv_v2.json'

        with open(path_to_json, 'w') as outfile:
            json.dump(self.final_dict, outfile)


def main():
    thing = Json_from_csv()

    path = r'/Users/Morten/Library/CloudStorage/OneDrive-SharedLibraries-ITU/Sofus Sebastian Schou Konglevoll - Bachelor/all_skeleton/all_skeleton_preprocessed_v3.csv'
    thing.dict_to_json(path)

if __name__ == '__main__':
    main()