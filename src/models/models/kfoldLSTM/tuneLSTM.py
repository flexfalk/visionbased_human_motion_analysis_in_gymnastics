
from torch import nn, optim
import torch.nn.functional as F
import torch.utils.data as data
import torch
import pandas as pd
from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np
from code.models import BI_LSTM as LSTM
from DataLoaderLSTM import LoadValTestJSON
from hyperparameterLSTM import hyperparameter_LSTM
import optuna
import json
from sklearn.metrics import f1_score, accuracy_score
from time import time
from code.models.BI_LSTM import LSTMTagger as LSTM
from sklearn.metrics import confusion_matrix

class tuner:

    def __init__(self):
        # self.validation_data = None
        # self.train_data = None
        self.model = LSTM(embedding_dim=66, hidden_dim=4, tagset_size=4)
        self.tag_to_ix = {"idle": 0, "take-off": 1, "skill": 2, "landing": 3}
        pass

    def tune_in_sub_folder(self, path_to_subfolder, key):


        path = path_to_subfolder

        # For Windows
        # path_train = path + "\\train.csv"
        # path_validation = path + "\\validation.csv"

        # For Mac
        path_train = path + "/train.json"
        path_validation = path + "/val.json"

        dataloader = LoadValTestJSON()

        self.validation_data, self.train_data = dataloader.LoadValAndTrain(val_path=path_validation, train_path=path_train)

        # print("self.train_data: ", type(self.train_data))

        study = optuna.create_study(direction="maximize")
        study.optimize(self.objective, n_trials=18)

        best_params = study.best_params

        # For Windows
        # with open(path + '\\best_params' + key +'.json', 'w') as fp:
        #     json.dump(best_params, fp)

        # For Mac
        with open(path + '/LSTM_best_params_v2_' + key + '.json', 'w') as fp:
            json.dump(best_params, fp)

        return best_params

    def get_predictions(self, ensemblelist):

        start = np.zeros(ensemblelist[0].shape)


        for array in ensemblelist:
            start += array

        preds = np.argmax(start, axis=1)

        return preds.tolist()

    def predict_ensemble(self, test_data, preds):

        y_list = []

        for i, (x,y) in enumerate(test_data):

            labels = self.model.prepare_sequence_Y(y, self.tag_to_ix)
            y_list += labels

        f1 = f1_score(y_list, preds, average='macro')
        acc = accuracy_score(y_list, preds)

        confs_matrix = confusion_matrix(y_list, preds)
        cm = confs_matrix.astype('float') / confs_matrix.sum(axis=1)[:, np.newaxis]

        return acc, f1, cm.tolist()

    def objective(self, trial):

        lr = trial.suggest_float("lr", 0.01, 0.15)
        momentum = trial.suggest_float("momentum", 0.1, 0.9)
        # beta1 = trial.suggest_float('beta1', 0.9, 0.9)
        # beta2 = trial.suggest_float('beta2', 0.999, 0.999)

        hyperLSTM = hyperparameter_LSTM()

        _, f1 = hyperLSTM.run(val_data=self.validation_data, train_data=self.train_data, epochs=20, lr=lr, momentum=momentum)

        return f1


    def do_everything(self, path_to_kfold, key):

        # path_to_kfold = "C:\Users\sofu0\OneDrive - ITU\Bachelor\kfold"

        folders = ["fold0", "fold1", "fold2", "fold3", "fold4"]
        subfolders = ["sub_folder0", "sub_folder1", "sub_folder2", "sub_folder3"]

        for folder in folders:
            overview_dict = {}
            ensemble_results ={}
            batman_results = {}
            lrs = []
            momentums = []
            # For Windows
            # path_to_folder = path_to_kfold + "\\" + folder

            # For Mac
            path_to_folder = path_to_kfold + "/" + folder


            # Create Test Set
            path_to_test = path_to_folder + '/tes.json'

            dataloader = LoadValTestJSON()

            test, _ = dataloader.LoadValAndTrain(val_path=path_to_test, train_path=path_to_test)

            ensemble_predictions = []

            for subfolder in subfolders:
                super_robin_stats = {}
                # For Windows
                # path_to_subfolder = path_to_folder + "\\" + subfolder

                # For Mac
                path_to_subfolder = path_to_folder + "/" + subfolder

                print(path_to_subfolder)

                q = self.tune_in_sub_folder(path_to_subfolder, key)

                lrs.append(q["lr"])
                momentums.append(q["momentum"])

                # train super robin
                super_robin = hyperparameter_LSTM()

                # For Windows
                # path_train = path_to_subfolder + "\\train.csv"

                # For Mac
                path_train = path_to_subfolder + "/train.json"

                dataloader = LoadValTestJSON()

                _, train = dataloader.LoadValAndTrain(val_path=path_train, train_path=path_train)

                # train = SkeletonData(path_train)
                # train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, sampler=None)
                s = time()
                super_robin_running_loss = super_robin.train(train_data=train, epochs=150, lr=q['lr'], momentum=q['momentum'])

                # For Windows
                # super_robin.save_model(path_to_subfolder + "\\CNN_super-robin")

                # For Mac
                super_robin.save_model(path_to_subfolder + "/LSTM_super-robin_v2")

                t = time()
                print("Time training a super robin at 150 epochs", t - s)


                # get softmax for using in ensemble
                softmax = super_robin.softmax(test)

                super_robin_stats["lr"] = q["lr"]
                super_robin_stats["momentum"] = q["momentum"]
                super_robin_stats["loss"] = super_robin_running_loss
                super_robin_stats["softmax_on_testset"] = softmax

                softmax = np.asarray(softmax).reshape((len(softmax), 4))

                # For Windows

                # with open(path_to_subfolder + '\\super_robin_stats' + key + '.json', 'w') as fp:
                #     json.dump(super_robin_stats, fp)

                # For Mac

                with open(path_to_subfolder + '/LSTM_super_robin_stats_v2_' + key + '.json', 'w') as fp:
                    json.dump(super_robin_stats, fp)

                print(softmax.shape)
                ensemble_predictions.append(softmax)

            preds = self.get_predictions(ensemble_predictions)

            acc, f1, confs_matrix = self.predict_ensemble(test, preds)



            print('ensemble Reuslts')
            print('Accuracy: ', acc)
            print('F1: ', f1)
            print()
            print('Confusion matrix: ', confs_matrix)

            ensemble_results["acc"] = acc
            ensemble_results["f1"] = f1
            ensemble_results['Confusion Matrix'] = confs_matrix

            # For Windows
            # with open(path_to_folder + '\\alfred_stats' + key + '.json', 'w') as fp:
            #     json.dump(ensemble_results, fp)

            # For Mac
            with open(path_to_folder + '/LSTM_alfred_stats_v2_' + key + '.json', 'w') as fp:
                json.dump(ensemble_results, fp)

            # save all parameters to an overview dict
            overview_dict["lr"] = lrs
            overview_dict["momentum"] = momentums

            # For Windows
            # with open(path_to_folder + '\\overview_parameters' + key + '.json', 'w') as fp:
            #     json.dump(overview_dict, fp)

            # For Mac
            with open(path_to_folder + '/LSTM_moverview_parameters_v2_' + key + '.json', 'w') as fp:
                json.dump(overview_dict, fp)

            # batman
            best_lr = np.mean(lrs)
            best_momentum = np.mean(momentums)

            # train batman

            # For Windows
            # path_train = path_to_folder + "\\train.csv"

            # For Mac
            path_train = path_to_folder + "/tra.json"

            dataloader = LoadValTestJSON()

            _, train = dataloader.LoadValAndTrain(val_path=path_train, train_path=path_train)

            # train = SkeletonData(path_train)
            # train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, sampler=None)

            batman = hyperparameter_LSTM()
            batman_running_loss = batman.train(train_data=train, epochs=150, lr=best_lr, momentum=best_momentum)

            acc, f1, confs_matrix = batman.check_accuracy(test)

            batman_results["acc"] = acc
            batman_results["f1"] = f1
            batman_results["loss"] = batman_running_loss
            batman_results['Confusion matrix'] = confs_matrix

            #
            # # For Windows
            # # with open(path_to_folder + '\\batman_results' + key + '.json', 'w') as fp:
            # #     json.dump(batman_results, fp)
            # #
            # batman.save_model(path_to_folder + "\\CNN_batman_model")


            # For Mac
            with open(path_to_folder + '/LSTM_batman_results_v2_' + key + '.json', 'w') as fp:
                json.dump(batman_results, fp)

            batman.save_model(path_to_folder + "/LSTM_batman_model_v2")


def main():

    #Before you run set 3 parameters;
    # epoch1 (for training the noobs) line 56. default=5,
    # epoch2 (for training the bigboy, line 108 default=10,
    # and n_trials for number of trials hyper parameter tuning line 40, default=5

    #instantiate tuner
    mytuner = tuner()

    #write metadata for naming.
    date = "11-05-2021"
    model = "LSTM"
    person = "Morten"
    key = date + model + person

    #the path to you kfold folder
    # path_to_k_fold = r"C:\Users\sofu0\OneDrive - ITU\Bachelor\kfold"
    path_to_k_fold = r'/Users/Morten/Library/CloudStorage/OneDrive-SharedLibraries-ITU/Sofus Sebastian Schou Konglevoll - Bachelor/kfold'

    #does everthing
    mytuner.do_everything(path_to_k_fold, key)


if __name__ == "__main__":
    main()















