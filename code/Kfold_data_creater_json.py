import os
from sklearn.model_selection import KFold
import pandas as pd
import random


def main():
    json_df = pd.read_json(
        r"C:\Users\Gustav Bakhauge\ITU\Sofus Sebastian Schou Konglevoll - Bachelor\all_skeleton\skeleton_from_csv_v2.json")

    json_good_vid_subset = json_df[["Video2_3init", "Video3_22init", "Video5_video9init", "Video11_Video10init", "Video13_13init"]]

    json_without_good_vid_subset = json_df.drop(["Video2_3init", "Video3_22init", "Video5_video9init", "Video11_Video10init", "Video13_13init"], axis=1)

    list_of_keys = json_without_good_vid_subset.keys().to_list()
    print(list_of_keys)
    random.shuffle(list_of_keys)
    print(list_of_keys)

    for i in range(5):
        chunk1 = list_of_keys[:40]
        chunk2 = list_of_keys[40:80]
        chunk3 = list_of_keys[80:120]
        chunk4 = list_of_keys[120:160]
        chunk5 = list_of_keys[160:]
        if i == 0:
            train_keys = chunk1 + chunk2 + chunk3 + chunk4
            test_keys = chunk5
        elif i == 1:
            train_keys = chunk2 + chunk3 + chunk4 + chunk5
            test_keys = chunk1
        elif i == 2:
            train_keys = chunk3 + chunk4 + chunk5 + chunk1
            test_keys = chunk2
        elif i == 3:
            train_keys = chunk4 + chunk5 + chunk1 + chunk2
            test_keys = chunk3
        elif i == 4:
            train_keys = chunk5 + chunk1 + chunk2 + chunk3
            test_keys = chunk4
        train = json_without_good_vid_subset[train_keys]
        test = json_without_good_vid_subset[test_keys]
        rootdir = r"C:\Users\Gustav Bakhauge\ITU\Sofus Sebastian Schou Konglevoll - Bachelor\kfold\fold" + str(i)
        test.to_json(rootdir + r"\tes.json")
        train.to_json(rootdir + r"\tra.json")
        print(rootdir)

if __name__=="__main__":
    main()