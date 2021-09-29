import numpy as np
import keras
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import RepeatVector
from keras.layers import TimeDistributed
import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def dropConfidence(df):
    drop_idx = list(range(2, df.shape[1], 3))
    df = df.drop(drop_idx, axis=1)
    return df

def getXcoords(df):
    drop_idx = list(range(1, df.shape[1], 2))
    df = df.drop(drop_idx, axis=1)
    return df

def getYcoords(df):
    drop_idx = list(range(0, df.shape[1], 2))
    df = df.drop(drop_idx, axis=1)
    return df


def getData():
    with open("keypoints5.json") as jsonFile:
        jsonObject1 = json.load(jsonFile)
        jsonFile.close()
        #print((len(jsonObject1)))
        df = pd.DataFrame.from_records(jsonObject1)
        df = dropConfidence(df)
        df.columns = np.arange(len(df.columns))
        #print(df)
        x = getXcoords(df)
        #print(x.loc[0])
        y = getYcoords(df)
        #print(df.shape[0])
        for i in range(df.shape[0]):
            #print("i:", i)
            fig, ax = plt.subplots()
            x_vals = x.loc[i]
            y_vals = y.loc[i]
            ax.plot(x_vals, y_vals, 'x')
            for j in range(25):
                plt.annotate(str(j), (x_vals.iloc[j], y_vals.iloc[j]))
            ax.set_ylim(ax.get_ylim()[::-1])
            plt.title('%s Plot' %i)
            #ax[1].invert_yaxis()
            #plt.show()
            #plt.savefig('plot1/%s Plot' %i)
            plt.close(fig)
    return df



