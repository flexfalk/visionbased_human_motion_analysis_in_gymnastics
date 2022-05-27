from preprocessing import PreProcesser
import pandas as pd


def main():
    pre = PreProcesser()
    df_list = []
    # df_all_data_skeleton = pd.read_csv(r"C:\Users\Gustav Bakhauge\ITU\Sofus Sebastian Schou Konglevoll - Bachelor\\all_skeleton\\all_skeleton_v2.csv")
    df_all_data_skeleton = pd.read_csv(r'/Users/Morten/Library/CloudStorage/OneDrive-SharedLibraries-ITU/Sofus Sebastian Schou Konglevoll - Bachelor/all_skeleton/all_skeleton_v2.csv')
    df_all_data_skeleton = df_all_data_skeleton.loc[df_all_data_skeleton["videoname"].isin(["Video1", "Video2", "Video3", "Video5", "Video11", "Video13", "Video17", "Video19"])]
    df_all_data_skeleton = pre.zero_remover(df_all_data_skeleton)
    videoname = df_all_data_skeleton.videoname.unique()

    for i in videoname:
        tmp_video_df = df_all_data_skeleton.loc[df_all_data_skeleton.videoname == i]
        clipname = tmp_video_df.clipname.unique()
        for j in clipname:
            tmp_clip_df = tmp_video_df.loc[tmp_video_df.clipname == j]
            # df = pre.normalize(tmp_clip_df)
            df_list.append(tmp_clip_df)

    tmp_df = pd.concat(df_list)
    final_df = tmp_df[['0x', '0y', '1x', '1y', '2x', '2y', '3x', '3y', '4x',
       '4y', '5x', '5y', '6x', '6y', '7x', '7y', '8x', '8y', '9x', '9y', '10x',
       '10y', '11x', '11y', '12x', '12y', '13x', '13y', '14x', '14y', '15x',
       '15y', '16x', '16y', '17x', '17y', '18x', '18y', '19x', '19y', '20x',
       '20y', '21x', '21y', '22x', '22y', '23x', '23y', '24x', '24y', '25x',
       '25y', '26x', '26y', '27x', '27y', '28x', '28y', '29x', '29y', '30x',
       '30y', '31x', '31y', '32x', '32y', 'finals', 'clipname', 'videoname']]

    final_df = pre.row_remover(final_df,"finals", "Unlabeled")
    labels = final_df.finals.unique()
    print(labels)
    print(final_df.keys())
    # final_df.to_csv(r"C:\Users\Gustav Bakhauge\ITU\Sofus Sebastian Schou Konglevoll - Bachelor\\all_skeleton\\all_skeleton_preprocessed_v3.csv", index=False)
    final_df.to_csv(r'/Users/Morten/Library/CloudStorage/OneDrive-SharedLibraries-ITU/Sofus Sebastian Schou Konglevoll - Bachelor/all_skeleton/all_skeleton_preprocessed_v3.csv', index=False)






if __name__=="__main__":
    main()
