import pandas as pd

path_to_raw_gt = '../../data/flight/raw_data/truth_sample.txt'
path_to_raw_data = '../../data/flight/raw_data/data.txt'


def cook_data():
    raw_gt = pd.read_csv(path_to_raw_gt, sep='\t', names=['O', 'V'], index_col=False).drop_duplicates()
    gt_data = raw_gt[raw_gt.V.notnull()]
    gt_data = gt_data.ix[gt_data.O.drop_duplicates().index]
    gt_data.to_csv('../../data/flight/data/gt.csv', index=False)

    data_list = []
    raw_data = pd.read_csv(path_to_raw_data, sep='\t', names=range(38), index_col=False, low_memory=False)
    for obj in raw_data.iterrows():
        obj_ind = obj[0]
        raw_votes = obj[1]
        votes = raw_votes[raw_votes.notnull()]
        for vote in votes.iteritems():
            s = vote[0]
            try:
                val = int(vote[1])
            except ValueError:
                val = vote[1]
            data_list.append([s, obj_ind, val])
    data = pd.DataFrame(data=data_list, columns=['S', 'O', 'V'])
    data.to_csv('../../data/flight/data/data.csv', index=False)

if __name__ == '__main__':
    cook_data()
