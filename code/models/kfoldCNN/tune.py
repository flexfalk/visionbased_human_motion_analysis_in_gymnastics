import warnings
warnings.filterwarnings("ignore")
from torch import nn, optim
import torch.nn.functional as F
import torch.utils.data as data
import torch
import pandas as pd
from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np
from CNN import CNN
from dataloader import SkeletonData
from hyperparameter import hyperparameter_CNN
import optuna
import json
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
from time import time
# from

class tuner:

    def __init__(self):
        pass

    def tune_in_sub_folder(self, path_to_subfolder, key):


        path = path_to_subfolder
        # path = r"C:\Users\sofu0\OneDrive - ITU\Bachelor\kfold\fold1\sub_folder0"

        # path = r"C:\Users\sofu0\OneDrive - ITU\Bachelor\kfold\fold0\sub_folder0
        path_train = path + "\\train_non_padded.csv"
        path_validation = path + "\\validation_non_padded.csv"
        train = SkeletonData(path_train)
        validation = SkeletonData(path_validation)
        batch_size = 64

        self.train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size,
                                                   sampler=None)

        self.validation_loader = torch.utils.data.DataLoader(validation, batch_size=batch_size,
                                                        sampler=None)

        study = optuna.create_study(direction="maximize")
        study.optimize(self.objective, n_trials=15)

        best_params = study.best_params


        return best_params


    def objective(self, trial):

        lr = trial.suggest_float("lr", 0.01, 0.1)
        momentum = trial.suggest_float("momentum", 0.1, 0.9)

        hyperCNN = hyperparameter_CNN()

        _, f1 = hyperCNN.run(lr, momentum, self.validation_loader, self.train_loader, epochs=25)

        return f1

    def get_predictions(self, ensemblelist):
        start = np.zeros(ensemblelist[0].shape)

        for arrays in ensemblelist:
            start += arrays

        # print(start)
        # print(start.shape)
        preds = np.argmax(start, axis=1)

        print(preds)
        print(type(preds))


        return preds.tolist()




    def predict_ensemble(self, test_loader, preds):

        y_list = []

        for i, (x, y) in enumerate(test_loader):

            y_list += y.tolist()

        f1 = f1_score(y_list, preds, average='macro')
        acc = accuracy_score(y_list, preds)

        confs_matrix = confusion_matrix(y_list, preds)
        cm = confs_matrix.astype('float') / confs_matrix.sum(axis=1)[:, np.newaxis].tolist()
        return acc, f1, cm.tolist()


    def do_everything(self, path_to_kfold, key):

        batch_size = 64

        # path_to_kfold = "C:\Users\sofu0\OneDrive - ITU\Bachelor\kfold"

        folders = ["fold0", "fold1", "fold2", "fold3", "fold4"]
        subfolders = ["sub_folder0", "sub_folder1", "sub_folder2", "sub_folder3"]

        for folder in folders:
            overview_dict = {}
            ensemble_results = {}
            batman_results = {}
            lrs = []
            momentums = []
            path_to_folder = path_to_kfold + "\\" + folder

            #create testset
            path_to_test = path_to_folder + "\\test_non_padded.csv"

            test = SkeletonData(path_to_test)

            test_loader = torch.utils.data.DataLoader(test, batch_size=batch_size, sampler=None)

            ensemble_predictions = []

            for subfolder in subfolders:
                super_robin_stats = {}
                path_to_subfolder = path_to_folder + "\\" + subfolder
                print(path_to_subfolder)

                q = self.tune_in_sub_folder(path_to_subfolder, key)

                lrs.append(q["lr"])
                momentums.append(q["momentum"])

                #train super robin
                super_robin = hyperparameter_CNN()
                path_train = path_to_subfolder + "\\train_non_padded.csv"
                train = SkeletonData(path_train)
                train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, sampler=None)
                s = time()
                super_robin_running_loss = super_robin.train(lr=q["lr"], momentum=q["momentum"], train_loader=train_loader, epochs=150)
                super_robin.save_model(path_to_subfolder + "\\"+ key + "CNN_super-robin")
                t = time()
                print("Time training a super robin at 150 epochs", t-s)


                #get softmax for using in ensemble
                softmax = super_robin.softmax(test_loader)

                super_robin_stats["learning_rate"] = q["lr"]
                super_robin_stats["momentum"] = q["momentum"]
                super_robin_stats["loss"] = super_robin_running_loss
                super_robin_stats["softmax_on_testset"] = softmax


                softmax = np.asarray(softmax).reshape((len(softmax), 4))

                with open(path_to_subfolder + '\\super_robin_stats' + key + '.json', 'w') as fp:
                    json.dump(super_robin_stats, fp)

                print(softmax.shape)
                ensemble_predictions.append(softmax)

            preds = self.get_predictions(ensemble_predictions)

            acc, f1, cm = self.predict_ensemble(test_loader, preds)

            ensemble_results["acc"] = acc
            ensemble_results["f1"] = f1
            ensemble_results["cm"] = cm

            with open(path_to_folder + '\\alfred_stats' + key + '.json', 'w') as fp:
                json.dump(ensemble_results, fp)


            #save all parameters to an overview dict
            overview_dict["lrs"] = lrs
            overview_dict["momentums"] = momentums

            with open(path_to_folder + '\\overview_parameters' + key + '.json', 'w') as fp:
                json.dump(overview_dict, fp)

            #batman
            best_lr = np.mean(lrs)
            best_momentum = np.mean(momentums)

            #train batman

            path_train = path_to_folder + "\\train_non_padded.csv"
            train = SkeletonData(path_train)
            train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size,sampler=None)

            batman = hyperparameter_CNN()
            batman_running_loss = batman.train(lr=best_lr,momentum=best_momentum,train_loader=train_loader,epochs=150)

            acc, f1, cm = batman.check_accuracy(test_loader)

            batman_results["acc"] = acc
            batman_results["f1"] = f1
            batman_results["cm"] = cm
            batman_results["loss"] = batman_running_loss


            with open(path_to_folder + '\\batman_results' + key + '.json', 'w') as fp:
                json.dump(batman_results, fp)

            batman.save_model(path_to_folder + "\\" + key +"CNN_batman_model")



def main():

    #Before you run set 3 parameters;
    # epoch1 (for training the noobs) line 56. default=5,
    # epoch2 (for training the bigboy, line 108 default=10,
    # and n_trials for number of trials hyper parameter tuning line 40, default=5

    start = time()

    #instantiate tuner
    mytuner = tuner()

    #write metadata for naming.
    date = "12-05-2021"
    model = "CNN"
    person = "sofus"
    key = date + model + person

    #the path to you kfold folder
    path_to_k_fold = r"C:\Users\sofu0\OneDrive - ITU\Bachelor\kfold"

    #does everthing
    mytuner.do_everything(path_to_k_fold, key)

    end = time()

    print(end-start)


if __name__ == "__main__":
    main()















