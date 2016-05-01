import pandas as pd
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
    golden_data = pd.DataFrame(data=[], columns=['R', 'V'])
    truth_obj_list = []
    obj_name_list = []
    with open('../data/restaurants/' + golden_file) as f:
        for line in f:
            params = line.strip().split('\t')
            rest_name = re.sub(r'\W+', '', params[0].lower())
            v = v_dict[params[1]]
            obj_name_list.append(rest_name)
            truth_obj_list.append(v)
    return [obj_name_list, truth_obj_list]


def get_data(obj_name_list, truth_obj_list):
    all_data = pd.read_csv('../data/restaurants/restaurants.csv')
    data_list = []
    for res_index, res_name in enumerate(obj_name_list):
        res_data = all_data[all_data.R == res_name]
        if res_data.empty:
            truth_obj_list[res_index] = None
            obj_name_list[res_index] = None
            continue
    truth_obj_list_new = [v for v in truth_obj_list if v != None]
    obj_name_list = [v for v in obj_name_list if v != None]
    for res_index, res_name in enumerate(obj_name_list):
        res_data = all_data[all_data.R == res_name]
        for psi in res_data.iterrows():
            psi = psi[1]
            data_list.append([int(psi.S), res_index, int(psi.V)])
    data = pd.DataFrame(data=data_list, columns=['S', 'O', 'V'])

    return [data, truth_obj_list_new]


def get_rest_data():
    # get_restaurants()
    obj_name_list, truth_obj_list = get_gt()
    data, truth_obj_list_new = get_data(obj_name_list=obj_name_list, truth_obj_list=truth_obj_list)

    return [data, truth_obj_list_new]
