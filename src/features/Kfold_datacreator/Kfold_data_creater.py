import os
from sklearn.model_selection import KFold
import pandas as pd


def main():

    data = pd.read_csv(r"C:\Users\Gustav Bakhauge\ITU\Sofus Sebastian Schou Konglevoll - Bachelor\\all_skeleton\\skeleton_csv_from_json_v2.csv")


    df_good_video_subset = data.loc[((data["videoname"] == "Video2") & (data["clipname"] == "3init"))|
                  ((data["videoname"] == "Video3") & (data["clipname"] == "22init"))|
                  ((data["videoname"] == "Video5") & (data["clipname"] == "video9init"))|
                  ((data["videoname"] == "Video11") & (data["clipname"] == "Video10init"))|
                  ((data["videoname"] == "Video13") & (data["clipname"] == "13init"))]

    some_index = data.loc[((data["videoname"] == "Video2") & (data["clipname"] == "3init"))|
                  ((data["videoname"] == "Video3") & (data["clipname"] == "22init"))|
                  ((data["videoname"] == "Video5") & (data["clipname"] == "video9init"))|
                  ((data["videoname"] == "Video11") & (data["clipname"] == "Video10init"))|
                  ((data["videoname"] == "Video13") & (data["clipname"] == "13init"))].index

    data_without_good_video_subset = data.drop(axis = 0, labels = some_index)

    # print(df_good_video_subset.shape)
    # print(df_good_video_subset["videoname"].unique())
    # print(data.shape)
    # print(data_without_good_video_subset.shape)
    #
    kf = KFold(n_splits=5, shuffle=True)
    count = 0
    for train_index, test_index in kf.split(data_without_good_video_subset):


        train, test = data_without_good_video_subset.iloc[train_index], data_without_good_video_subset.iloc[test_index]

        folder_path = r"C:\Users\Gustav Bakhauge\ITU\Sofus Sebastian Schou Konglevoll - Bachelor\kfold\fold" + str(count)
        print(folder_path)
        try:
            os.mkdir(folder_path)
        except OSError as error:
            pass
        # print(data_without_good_video_subset.shape)
        # print("Size: ", train.shape, test.shape)
        train = train.sample(frac=1)
        test = test.sample(frac=1)
        train.to_csv(folder_path + r"\train.csv", index=False)
        test.to_csv(folder_path + r"\test.csv", index=False)
        count +=1
    df_good_video_subset.to_csv(r"C:\Users\Gustav Bakhauge\ITU\Sofus Sebastian Schou Konglevoll - Bachelor\kfold\test_good_video_subset.csv")





if __name__=="__main__":
    main()

