import pandas as pd
import re


def get_restaurants():
    # k = pd.read_csv('restaurants.csv')
    # d = sorted(list(k.R.drop_duplicates()))
    # df = pd.DataFrame(data=d)
    # df.to_csv('../../data/restaurants/rest_name')

    rest_files = [
        'restaurants_2009_1_22.txt',
        'restaurants_2009_1_29.txt',
        'restaurants_2009_2_05.txt',
        'restaurants_2009_2_12.txt',
        'restaurants_2009_2_19.txt',
        'restaurants_2009_2_26.txt',
        'restaurants_2009_3_05.txt',
        'restaurants_2009_3_12.txt']

    source_name = sorted(['MenuPages', 'TasteSpace', 'NYMag', 'NYTimes', 'ActiveDiner', 'TimeOut',
                          'SavoryCities', 'VillageVoice', 'FoodBuzz', 'NewYork', 'OpenTable', 'DiningGuide'])

    data = pd.DataFrame(data=[], columns=['S', 'R', 'V'])
    data_list = []
    for f_name in rest_files:
        with open('../../data/restaurants/' + f_name) as f:
            for line in f:
                params = line.strip().split('\t')
                if len(params) < 2:
                    continue
                source = re.sub(r'\W+', '', params[0])
                if '(CLOSED)' in params[1]:
                    rest_name = params[1].replace('(CLOSED)', '').strip().lower()
                    val = 0
                else:
                    val = 1
                    rest_name = params[1].lower()
                rest_name = re.sub(r'\W+', '', rest_name)
                if rest_name == '':
                    continue

                data_list.append([source, rest_name, val])
    raw_data = pd.DataFrame(data=data_list, columns=['S', 'R', 'V'])

    for r in raw_data.R.drop_duplicates():
        d_r = raw_data[raw_data.R == r].drop_duplicates(subset='S', keep='last')
        if len(d_r) > 1:
            data = data.append(d_r)
    data.to_csv('../../data/restaurants/restaurants.csv')
    print len(data.R.drop_duplicates())


if __name__ == '__main__':
    get_restaurants()
