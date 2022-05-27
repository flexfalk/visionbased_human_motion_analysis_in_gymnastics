import os
from sklearn.model_selection import KFold
import pandas as pd

def main():
    kf = KFold(n_splits=4)
    rootdir = r"C:\Users\Gustav Bakhauge\ITU\Sofus Sebastian Schou Konglevoll - Bachelor\kfold"
    some_counter = 0
    for subdir, dirs, files in os.walk(rootdir):

        print(subdir, dirs, files)
        print(len(dirs))
        print(subdir[-5:-1])
        print(files[-1])

        if len(dirs) == 0 and subdir[-5:-1] == "fold":

            data = pd.read_csv(subdir + "\\" + files[-1])
            count=0
            for train_index, validation_index in kf.split(data):
                train, val = data.iloc[train_index], data.iloc[validation_index]
                folder_path = subdir + "\\sub_folder" + str(count)
                print(folder_path)
                try:
                    os.mkdir(folder_path)
                except OSError as error:
                    pass
                train.to_csv(folder_path + r"\train.csv", index=False)
                val.to_csv(folder_path + r"\validation.csv", index=False)

                count += 1


if __name__=="__main__":
    main()

