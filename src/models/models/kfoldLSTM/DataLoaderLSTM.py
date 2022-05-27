import json


class LoadValTestJSON:

    def __init__(self):
        pass

    def LoadJsonToList(self, path):

        file = open(path)
        dict = json.load(file)

        data = []

        for key in dict.keys():
            data.append((dict[key]['0'], dict[key]['1']))


        return data

    def LoadValAndTrain(self, val_path, train_path):

        val_set = self.LoadJsonToList(val_path)
        train_set = self.LoadJsonToList(train_path)


        return val_set, train_set

