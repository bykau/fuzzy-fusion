import numpy as np
import pandas as pd
import operator
import re


rest_files = [
        'restaurants_2009_1_22.txt',
        'restaurants_2009_1_29.txt',
        'restaurants_2009_2_05.txt',
        'restaurants_2009_2_12.txt',
        'restaurants_2009_2_19.txt',
        'restaurants_2009_2_26.txt',
        'restaurants_2009_3_05.txt',
        'restaurants_2009_3_12.txt']

golden_file = 'restaurants_golden.txt'

source_name = {'MenuPages': 0,
               'TasteSpace': 1,
               'NYMag': 2,
               'NYTimes': 3,
               'ActiveDiner': 4,
               'TimeOut': 5,
               'SavoryCities': 6,
               'VillageVoice': 7,
               'FoodBuzz': 8,
               'NewYork': 9,
               'OpenTable': 10,
               'DiningGuide': 11}

v_dict = {'Y': 1,
          'N': 0}


def get_restaurants():
    data = pd.DataFrame(data=[], columns=['S', 'R', 'V'])
    data_list = []
    for f_ind, f_name in enumerate(rest_files):
        with open('../../data/restaurants/' + f_name) as f:
            for line in f:
                params = line.strip().split('\t')
                if len(params) < 2:
                    continue
                source = source_name[re.sub(r'\W+', '', params[0])]
                if '(CLOSED)' in params[1]:
                    rest_name = params[1].replace('(CLOSED)', '').strip().lower()
                    val = 0
                else:
                    val = 1
                    rest_name = params[1].lower()
                rest_name = re.sub(r'\W+', '', rest_name)
                if rest_name == '':
                    continue

                data_list.append([source, rest_name, val, f_ind])
    raw_data = pd.DataFrame(data=data_list, columns=['S', 'R', 'V', 'F'])
    raw_data.to_csv('../../data/restaurants/raw_restaurants.csv')

    for r in raw_data.R.drop_duplicates():
        d_r = raw_data[raw_data.R == r].drop_duplicates(subset='S', keep='last')
        if len(d_r) > 1:
            for r in d_r.iterrows():
                if r[1].F != 7:
                    d_r.at[r[0], 'V'] = 0
            data = data.append(d_r)
    data.to_csv('../../data/restaurants/restaurants.csv')
    print len(data.R.drop_duplicates())
    print len(data[data.V == 0].R.drop_duplicates())


def get_gt():
    truth_val_list = []
    obj_name_list = []
    gt_data = {}
    with open('../data/restaurants/' + golden_file) as f:
        for line in f:
            params = line.strip().split('\t')
            rest_name = re.sub(r'\W+', '', params[0].lower())
            v = v_dict[params[1]]
            gt_data.update({rest_name: v})

    sorted_gt_data = sorted(gt_data.items(), key=operator.itemgetter(0))
    for gt in sorted_gt_data:
        obj_name_list.append(gt[0])
        truth_val_list.append(gt[1])

    return [obj_name_list, truth_val_list]


def get_data(obj_name_list, truth_val_list):
    all_data = pd.read_csv('../data/restaurants/restaurants.csv')
    data_list = []

    names = all_data.R.drop_duplicates()
    gt_names_list = sorted(list(set(names) & set(obj_name_list)))
    rest_names = list(set(names) - set(gt_names_list))
    all_names = gt_names_list + rest_names

    truth_val_list_new = []
    for v_ind, v in enumerate(obj_name_list):
        if v in gt_names_list:
            truth_val_list_new.append(truth_val_list[v_ind])

    for res_index, res_name in enumerate(all_names):
        res_data = all_data[all_data.R == res_name]
        for psi in res_data.iterrows():
            psi = psi[1]
            data_list.append([int(psi.S), res_index, int(psi.V)])
    data = pd.DataFrame(data=data_list, columns=['S', 'O', 'V'])

    return [data, truth_val_list_new]


def get_rest_data():
    # get_restaurants()
    obj_name_list, truth_val_list = get_gt()
    data, truth_val_list_new = get_data(obj_name_list=obj_name_list, truth_val_list=truth_val_list)

    return [data, truth_val_list_new]


def generate_swaps(data, pi):
    gt_len = 414
    swp_gt = 0
    numb_of_swaps = 0
    for psi in data.iterrows():
        psi_ind = psi[0]
        psi = psi[1]
        if np.random.binomial(1, pi, 1)[0] == 0:
            if len(data[data.O == psi.O]) <= 2:
                continue
            if psi_ind < gt_len:
                swp_gt += 1
            obj_ind = psi.O
            if obj_ind % 2 == 0:
                swp_ind = obj_ind + 1
            else:
                swp_ind = obj_ind - 1
            data.at[psi_ind, 'O'] = swp_ind
            numb_of_swaps += 1

    print 'total swaps: {}'.format(numb_of_swaps)
    print 'total swaps in %: {}'.format(float(numb_of_swaps)/len(data))
    print 'gt objects: {}'.format(gt_len)
    print 'swaps in gt clusters in %: {}'.format(float(swp_gt)/gt_len)
    print 'observations: {}'.format(len(data))
    print 'sources: 12'
    print 'objects: {}'.format(len(data.O.drop_duplicates()))
    return data
