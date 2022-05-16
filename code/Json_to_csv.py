import json
import pandas as pd

class json_to_csv:

    def __init__(self, path_to_json):
        self.path_to_json = path_to_json

    def read_json(self):

        skeleton_json = open(self.path_to_json)
        full_skeleton = json.load(skeleton_json)

        return full_skeleton

    def make_df(self):

        df = pd.DataFrame()

        full_skeleton = self.read_json()

        for key in full_skeleton.keys():
            print(key)

            for j in range(len(full_skeleton[key][0])):

                data_point = full_skeleton[key][0][j]
                label = full_skeleton[key][1][j]

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


    def df_to_csv(self, path):

        df = self.make_df()

        df.to_csv(path, index=False)



def main():

    path_to_new_csv = r'/Users/Morten/Library/CloudStorage/OneDrive-SharedLibraries-ITU/Sofus Sebastian Schou Konglevoll - Bachelor/all_skeleton/skeleton_csv_from_json_v2.csv'
    path = r'/Users/Morten/Library/CloudStorage/OneDrive-SharedLibraries-ITU/Sofus Sebastian Schou Konglevoll - Bachelor/all_skeleton/skeleton_good_padded_videos_v2.json'

    jm = json_to_csv(path)
    jm.df_to_csv(path_to_new_csv)

if __name__ == '__main__':
    main()