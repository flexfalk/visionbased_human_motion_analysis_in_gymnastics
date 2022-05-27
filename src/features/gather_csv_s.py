import pandas as pd
import os
# from preprocessing import PreProcesser

class gather_csv_s():

    def __init__(self):
        pass



    def merge_csv_and_label(self, path_to_skeleton, path_to_labels):

        df1 = pd.read_csv(path_to_skeleton)
        df2 = pd.read_csv(path_to_labels)
        df1["finals"] = df2["final"]

        return df1


    def do_stuff_in_folder(self, path_to_folder):

        df = pd.DataFrame()
        counter = len(os.listdir(path_to_folder))
        i = 0
        while i < counter:

            # print(i)
            subfolder = os.listdir(path_to_folder)[i]

            # For Windows
            # path_to_sub_sub_folder = path_to_folder + r'\\' + subfolder

            # For MAC
            path_to_sub_sub_folder = path_to_folder + r'/' + subfolder
            if subfolder[0] != '.':
                for file in os.listdir(path_to_sub_sub_folder):
                    # print(file)
                    if file == 'majorityVote_clean.csv':
                        # print("found majorVote")
                        labels = file

                    try:
                        if file.split("_")[1] == "skeleton-v2.csv":
                            # print("found X")
                            X = file

                    except IndexError:
                        continue


                # For Windows
                # path_to_labels = path_to_sub_sub_folder + r"\\" + labels
                # path_to_X = path_to_sub_sub_folder + r"\\" + X

                # For MAC
                path_to_labels = path_to_sub_sub_folder + r"/" + labels
                path_to_X = path_to_sub_sub_folder + r"/" + X


                some_df = pd.read_csv(path_to_X)
                some_df = some_df.drop(["Unnamed: 0"], axis= 1)
                try:
                    some_df = some_df.drop(["Unnamed: 0.1"], axis=1)
                except KeyError:
                    pass
                # print("#############")
                # print(some_df.keys())
                # print("#############")
                # pre = PreProcesser()
                # some_pre_df = pre.normalize(some_df)

                # For Windows
                # path_to_pp = path_to_sub_sub_folder + r"\\full_skeleton_v2.csv"

                # For MAC
                path_to_pp = path_to_sub_sub_folder + r"/full_skeleton_v2.csv"


                # some_pre_df.to_csv(path_to_pp)


                new_df = self.merge_csv_and_label(path_to_X, path_to_labels)
                new_df["clipname"] = X.split("_")[0]
                new_df["videoname"] = path_to_folder.split("/")[-1]
                #
                df = pd.concat([df, new_df])

                some_df.to_csv(path_to_pp, index=False)

                i += 1
            else:
                i += 1


        # print('you are done :)')

        return df



def main():

    cs = gather_csv_s()

    df = pd.DataFrame()

    for i in range(1, 20):

        # print(i)
        # q = cs.do_stuff_in_folder(r'C:\Users\Gustav Bakhauge\ITU\Sofus Sebastian Schou Konglevoll - Bachelor\clean_video\Video' + str(i))
        q = cs.do_stuff_in_folder(r'/Users/Morten/Library/CloudStorage/OneDrive-SharedLibraries-ITU/Sofus Sebastian Schou Konglevoll - Bachelor/clean_video/Video' + str(i))
        df = pd.concat([df, q])
        # print("you are done")
        print(df)

    # df.to_csv(r'C:\Users\sofu0\OneDrive - ITU\Bachelor\all_skeleton\all_skeleton_v2.csv', index=False)
    df = df.drop(columns=['Unnamed: 0'])
    df.to_csv(r'/Users/Morten/Library/CloudStorage/OneDrive-SharedLibraries-ITU/Sofus Sebastian Schou Konglevoll - Bachelor/all_skeleton/all_skeleton_v2.csv', index=False)

if __name__ == "__main__":
    main()

