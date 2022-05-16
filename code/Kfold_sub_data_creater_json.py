import os
import pandas as pd

rootdir = r"C:\Users\Gustav Bakhauge\ITU\Sofus Sebastian Schou Konglevoll - Bachelor\kfold"
count = 0
for subdir, dirs, files in os.walk(rootdir):
    if subdir[-5:-1] == "fold":

        json_train = pd.read_json(rootdir+"\\fold"+str(count)+"\\tra.json")
        list_train_keys = json_train.keys().to_list()
        print(len(list_train_keys))

        chunk1 = list_train_keys[:40]
        chunk2 = list_train_keys[40:80]
        chunk3 = list_train_keys[80:120]
        chunk4 = list_train_keys[120:]
        count += 1
        sub_count = 0

    if subdir[-11:-1] == "sub_folder":
        # print(sub_count)
        if sub_count == 0:
            train_keys = chunk1 + chunk2 + chunk3
            test_keys = chunk4
            train = json_train[train_keys]
            val = json_train[test_keys]
            val.to_json(subdir + r"\val.json")
            train.to_json(subdir + r"\train.json")
        elif sub_count == 1:
            train_keys = chunk4 + chunk1 + chunk2
            test_keys = chunk3
            train = json_train[train_keys]
            val = json_train[test_keys]
            val.to_json(subdir + r"\val.json")
            train.to_json(subdir + r"\train.json")
        elif sub_count == 2:
            train_keys = chunk3 + chunk4 + chunk1
            test_keys = chunk2
            train = json_train[train_keys]
            val = json_train[test_keys]
            val.to_json(subdir + r"\val.json")
            train.to_json(subdir + r"\train.json")
        elif sub_count == 3:
            train_keys = chunk2 + chunk3 + chunk4
            test_keys = chunk1
            train = json_train[train_keys]
            val = json_train[test_keys]

            val.to_json(subdir + r"\val.json")
            train.to_json(subdir + r"\train.json")

        sub_count += 1

