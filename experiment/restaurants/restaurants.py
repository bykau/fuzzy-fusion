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


if __name__ == '__main__':
    get_restaurants()
