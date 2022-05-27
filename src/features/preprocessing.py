import pandas as pd


class PreProcesser():

    def __init__(self):
        self.xlist = ['0x', '1x', '2x', '3x', '4x', '5x', '6x', '7x', '8x', '9x', '10x', '11x', '12x', '13x', '14x',
                      '15x', '16x', '17x', '18x', '19x', '20x', '21x', '22x', '23x', '24x', '25x', '26x', '27x', '28x',
                      '29x', '30x', '31x', '32x']
        self.ylist = ['0y', '1y', '2y', '3y', '4y', '5y', '6y', '7y', '8y', '9y', '10y', '11y', '12y', '13y', '14y',
                      '15y', '16y', '17y', '18y', '19y', '20y', '21y', '22y', '23y', '24y', '25y', '26y', '27y', '28y',
                      '29y', '30y', '31y', '32y']

    def row_remover(self, dataframe, key, identifier):
        df = dataframe.loc[dataframe[key] != identifier]
        return df

    def zero_remover(self,dataframe):
        for ix, item in dataframe.iterrows():
            T = item[self.xlist + self.ylist]
            if sum(T.abs()) == 0:
                dataframe = dataframe.drop(ix)
        return dataframe

    def normalize(self,dataframe, x_bool = True, y_bool = True):
        df = dataframe
        if x_bool:
            df = self._normalize_x(df)

        if y_bool:
            df = self._normalize_y(df)
        dataframe = df
        return dataframe

    def _normalize_x(self, dataframe):
        df = dataframe[self.xlist]
        hip_col = df["23x"]
        df = df.sub(hip_col, axis=0)
        dataframe.loc[:, tuple(self.xlist)] = df.loc[:,tuple(self.xlist)]
        return dataframe

    def _normalize_y(self, dataframe):
        df = dataframe[self.ylist]
        hip_val = df["23y"].iloc[0]
        df = df.sub(hip_val)
        dataframe.loc[:, tuple(self.ylist)] = df.loc[:,tuple(self.ylist)]
        return dataframe


    def normalize_live(self, dataframe, first_frame, x_bool = True, y_bool=True):
        df = dataframe

        if x_bool:
            df = self._normalize_x(df)

        if y_bool:
            df = self._normalize_y_live(df, first_frame)

        dataframe = df
        return dataframe

    def _normalize_y_live(self, dataframe, first_frame):
        # print(first_frame)
        df = dataframe[self.ylist]
        hip_val = first_frame["23y"].iloc[0]
        # print(hip_val)
        # hip_val = 182
        # print(hip_val)
        # hip_val = df["23y"].iloc[0]
        df = df.sub(hip_val)
        dataframe.loc[:, tuple(self.ylist)] = df.loc[:,tuple(self.ylist)]
        return dataframe







class LivePreprocessor():

    def __init__(self):
        self.xlist = ['0x', '1x', '2x', '3x', '4x', '5x', '6x', '7x', '8x', '9x', '10x', '11x', '12x', '13x', '14x',
                      '15x', '16x', '17x', '18x', '19x', '20x', '21x', '22x', '23x', '24x', '25x', '26x', '27x', '28x',
                      '29x', '30x', '31x', '32x']
        self.ylist = ['0y', '1y', '2y', '3y', '4y', '5y', '6y', '7y', '8y', '9y', '10y', '11y', '12y', '13y', '14y',
                      '15y', '16y', '17y', '18y', '19y', '20y', '21y', '22y', '23y', '24y', '25y', '26y', '27y', '28y',
                      '29y', '30y', '31y', '32y']

    def normalize(self, array, start_frame):

        #normalize x
        array[:, :, 0, :] = array[:, :, 0, :] - start_frame[:, :, 0, 23]

        #noramlzie y
        array[:, :, 1, :] = array[:,:,1, :] - array[:, :, 1, 23]
        return array








