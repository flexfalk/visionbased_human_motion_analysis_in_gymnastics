import pandas as pd
import os


class VoteCleaner:

    def __init__(self):
        pass

    def read_csv(self, csv_path):
        df = pd.read_csv(csv_path)
        return df

    def clean_vote(self, csv_path) -> pd.DataFrame:
        df = self.read_csv(csv_path)
        final_df = pd.DataFrame()

        prev_step = None
        action_checkpoint = None
        take_off = False

        for i in range(len(df)):
            line = df[i:i+1]
            status = line['final']

            if take_off and status[i] == 'idle':
                print(action_checkpoint)
                for j in range(action_checkpoint, i):
                    # final_df[j:j+1]['final'] = 'idle'
                    final_df.loc[j, 'final'] = 'idle'
                    print(j)
                final_df = pd.concat([final_df, line])
            else:
                final_df = pd.concat([final_df, line])

            if status[i] == 'take-off' and prev_step != 'take-off':
                action_checkpoint = i
                take_off = True

            if status[i] == 'skill':
                take_off = False

            prev_step = status[i]
        return final_df

    def run(self, csv_path) -> None:
        df = self.clean_vote(csv_path)
        path = csv_path[:-4] + '_clean.csv'
        df.to_csv(path)

    def annotate_folder(self, path_to_folder):

        counter = len(os.listdir(path_to_folder))
        i = 0
        while i < counter:

            print(i)

            subfolder = os.listdir(path_to_folder)[i]
            path_to_sub_sub_folder = path_to_folder + r'/' + subfolder

            if subfolder[0] != '.':
                for file in os.listdir(path_to_sub_sub_folder):
                    if file[:3] == 'maj':
                        csv_name = file
                    else:
                        pass

                path_to_csv = path_to_sub_sub_folder + r"/" + csv_name

                print(path_to_csv)
                i += 1
                df = self.clean_vote(path_to_csv)
                csv_path = path_to_csv[:-4] + '_clean.csv'
                df.to_csv(csv_path)
            else:
                i += 1


def main():
    VC = VoteCleaner()

    for i in range(1, 20):
        path = r'/Users/Morten/Library/CloudStorage/OneDrive-SharedLibraries-ITU/Sofus Sebastian Schou Konglevoll - Bachelor/clean_video' + '/Video' + str(i)
        VC.annotate_folder(path)


if __name__ == "__main__":
    main()