import pandas as pd
import re
from os import listdir
from os.path import isfile, join
from datetime import timedelta


path_to_raw_gt = '../../data/flight/raw_data/truth_sample.txt'
path_to_raw_data = '../../data/flight/raw_data/data.txt'


def gt():
    raw_gt = pd.read_csv(path_to_raw_gt, sep='\t', names=['O', 'V'], index_col=False).drop_duplicates()
    gt_data = raw_gt[raw_gt.V.notnull()]
    gt_data.to_csv('../../data/flight/data/gt.csv', index=False)


if __name__ == '__main__':
    gt()
