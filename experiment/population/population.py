import pandas as pd


path_to_raw_data = '../../data/population/raw_data/popTuples.txt'
path_to_raw_gt = '../../data/population/raw_data/popAnswersOut.txt'


def get_population():
    # data = pd.read_csv(path_to_raw_data, sep='\t', header=None)[[4, 0, 6, 1, 7]]
    #
    # data[0] = data[0].map(lambda x: x.lower())
    # data[1] = data[6].map(lambda x: x[5:9])
    #
    # l = []
    #
    # def to_datetime(x):
    #     try:
    #         a = pd.to_datetime(x[1:21])
    #         l.append(a)
    #         return a
    #     except:
    #         print x[1:21]
    #         return x[1:21]
    #
    # data[6] = data[6].map(to_datetime)
    # data = data[data[6].isin(l)]
    # data = data.sort_values(by=6).drop_duplicates(subset=[4, 0, 1],  keep='last')
    # data = data.sort_values(by='O')
    # obj_name = data['O'] +', ' + data['Y']
    # data = pd.DataFrame(zip(data['S'], obj_name.values, data['V'].values), columns=['S', 'O', 'V'])
    data = pd.read_csv('../../data/population/data/pop_data.csv')
    s_index = range(len(data['S'].drop_duplicates()))
    sources = sorted(data['S'].drop_duplicates())
    s_dict = dict(zip(sources, s_index))
    data['S'] = data['S'].map(lambda x: s_dict[x])

    data.to_csv('../../data/population/data/pop_data.csv')


def get_gt():
    gt_raw_data = pd.read_csv(path_to_raw_gt, header=None)
    gt_raw_data[2] = gt_raw_data[2].map(lambda x: str(x).strip())

    obj_name = gt_raw_data[0] + ',' + gt_raw_data[1] + ', ' + gt_raw_data[2]
    gt_data = pd.DataFrame(zip(obj_name.values, gt_raw_data[3].values), columns=['O', 'V'])

    gt_data.to_csv('../../data/population/data/pop_gt.csv')


if __name__ == '__main__':
    get_population()
    # get_gt()
