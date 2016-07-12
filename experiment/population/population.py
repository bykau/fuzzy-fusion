import pandas as pd
import re
from os import listdir
from os.path import isfile, join
from datetime import timedelta
import math


path_to_raw_data = '../../data/population/raw_data/popTuples.txt'
path_to_raw_gt = '../../data/population/raw_data/popAnswersOut.txt'


def get_population():
    data = pd.read_csv(path_to_raw_data, sep='\t', header=None)[[4, 0, 7]]
    data = data.drop_duplicates(subset=[0, 4], keep='last')

    def lower(x):
        return x.lower()
    data[0] = data[0].map(lower)

    data.to_csv('pop_data.csv')


def get_gt():
    gt_raw_data = pd.read_csv(path_to_raw_gt, header=None)
    obj_name = gt_raw_data[0] + ',' + gt_raw_data[1]
    gt_data = pd.DataFrame(zip(obj_name.values, gt_raw_data[3].values), columns=['O', 'V'])

    gt_data.to_csv('../../data/population/data/pop_gt.csv')


if __name__ == '__main__':
    # get_population()
    get_gt()
