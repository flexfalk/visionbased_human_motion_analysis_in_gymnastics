import json
import pandas as pd

path= r'/Users/Morten/Library/CloudStorage/OneDrive-SharedLibraries-ITU/Sofus Sebastian Schou Konglevoll - Bachelor/kfold/fold0/sub_folder0/train.json'

class jsonPadCSV:

    def __init__(self):
        pass

    def read_json(self, path_to_json):
        skeleton_json = open(path_to_json)
        full_skeleton = json.load(skeleton_json)

        return full_skeleton


    def make_df(self, path):
        df = pd.DataFrame()

        full_skeleton = self.read_json(path)

        for key in full_skeleton.keys():
            print(key)

            data_point = full_skeleton[key]['0']
            label = full_skeleton[key]['1']

            x_values = []
            y_values = []

            for i in range(0, 66, 2):
                x_values.append(data_point[i])

            for i in range(1, 66, 2):
                y_values.append(data_point[i])
            d = {}
            # print(len(frame))

            for i in range(33):
                d[str(i) + "x"] = [x_values[i]]
                d[str(i) + "y"] = [y_values[i]]

            d['finals'] = label
            names = key.split('_')
            d['clipname'] = names[1]
            d['videoname'] = names[0]

            dframe = pd.DataFrame(data=d)
            df = pd.concat([df, dframe])

        return df


loader = jsonPadCSV()

df = loader.make_df(path)

print(df)



# def sub_folder_creator(path):
#
#
#
#
#
# def main():


    # path =