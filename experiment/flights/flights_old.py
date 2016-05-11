import pandas as pd
import re
from os import listdir
from os.path import isfile, join
from datetime import timedelta
import math


path_to_data = '../../data/flight/clean_flight/'
path_to_raw_data = '../../data/flight/raw_flight/'
path_to_gt = '../../data/flight/flight_truth/'
path_to_raw_gt = '../../data/flight/raw_gt/'
t_delta = 10.


def raw_flights():
    data_list = []
    flight_files = [f for f in listdir(path_to_data) if isfile(join(path_to_data, f))][1:]

    for f_name in flight_files:
        # with open('../../data/flight/clean_flight/' + f_name) as f:
        with open('../../data/flight/fl_data/data.txt') as f:
            for line in f:
                params = line.strip().split('\t')
                data_ind = [2, 3, 5, 6]
                for i in data_ind:
                    if i not in range(len(params)):
                        continue
                    param = params[i]
                    if param:
                        param = param.split('(', 1)[0]
                        param = param.replace('Not Available', '')\
                            .replace('AM', 'AM ').replace('PM', 'PM ')\
                            .replace('noon', 'pm')\
                            .replace('aJan', 'a Jan').replace('Contact Airline', ' ')\
                            .replace('On Time', '').replace('Delayed', '')\
                            .replace('Cancelled', '').replace('&nbsp', ' ')\
                            .replace('*', ' ').replace('Canceled', '')\
                            .replace('result unknown', '').replace('midnight', 'am')\
                            .replace('Possible delay pending airline decision.', '')\
                            .replace('Rerouted', '').replace('Extra Stop', '')\
                            .replace('Pending will update at', ' ').replace('?', '').replace('$', ' ')
                        param = re.sub(' [0-9]+/[0-9]+ ', ' ', param)
                        if 'pm' not in param:
                            param = param.replace('p', 'p ')
                        if 'am' not in param and 'Sat' not in param and 'Jan' not in param:
                            param = param.replace('a', 'a ')
                        if '2011' not in param and '/11 ' not in param and '-11' not in param\
                                and '2012' not in param and '/12 ' not in param and '-12' not in param:
                            if '2012' in f_name:
                                param += ' 2012'
                            else:
                                param += ' 2011'
                        param = param.strip()
                        params[i] = pd.to_datetime(param, infer_datetime_format=True)
                data_list.append(params)
    raw_data = pd.DataFrame(data=data_list, columns=['S', 'F', 'SD', 'AD', 'DG', 'SA', 'AA', 'AG'])
    raw_data.to_csv('../../data/flight/cleaned_data/raw_data.csv')


def raw_gt():
    gt_files = [f for f in listdir(path_to_gt) if isfile(join(path_to_gt, f))]
    data_list = []
    for f_name in gt_files:
        with open('../../data/flight/flight_truth/' + f_name) as f:
            for line in f:
                params = line.strip().split('\t')
                date_ind = [1, 2, 4, 5]
                for i in date_ind:
                    param = params[i]
                    if param:
                        param = param.split('(', 1)[0]
                        param = param.replace('Not Available', '')\
                            .replace('AM', 'AM ').replace('PM', 'PM ')\
                            .replace(' 11/30 ', ' ').replace('noon', 'pm')\
                            .replace('aJan', 'a Jan').replace('Delayed', '')\
                            .replace('On Time', ' ').replace('Cancelled', '')\
                            .replace('?', '').replace('$', ' ')
                        param = re.sub(' 12/[0-9]+ ', ' ', param)
                        if 'pm' not in param:
                            param = param.replace('p', 'p ')
                        if 'am' not in param and 'Sat' not in param and 'Jan' not in param:
                            param = param.replace('a', 'a ')
                        if '2011' not in param and '/11 ' not in param and '-11' not in param\
                                and '2012' not in param and '/12 ' not in param:
                            if '2012' in f_name:
                                param += ' 2012'
                            else:
                                param += ' 2011'
                        param = param.strip()
                        params[i] = pd.to_datetime(param, infer_datetime_format=True)
                data_list.append(params)
    gt_temp = pd.DataFrame(data=data_list, columns=['F', 'SD', 'AD', 'DG', 'SA', 'AA', 'AG'])
    gt_temp.to_csv('../../data/flight/cleaned_data/raw_gt.csv')


def gt():
    data_list = []
    attr_list = ['SD', 'AD', 'DG', 'SA', 'AA', 'AG']
    raw_gt = pd.read_csv('../../data/flight/cleaned_data/raw_gt.csv', index_col=0)
    for flight in raw_gt.iterrows():
        flight = flight[1]
        for attr in attr_list:
            if type(flight[attr]) is not float:
                if flight[attr].strip():
                    if type(flight.SD) is not float:
                        date_f = flight.SD[0:10]
                    elif type(flight.AD) is not float:
                        date_f = flight.AD[0:10]
                    elif type(flight.SA) is not float:
                        date_f = flight.SA[0:10]
                    elif type(flight.AA) is not float:
                        date_f = flight.AA[0:10]
                    obj_name = date_f + '-' + flight.F + '-' + attr
                    data_list.append([obj_name, flight[attr]])
    gt_data = pd.DataFrame(data=data_list, columns=['O_name', 'V']).drop_duplicates(subset='O_name')
    o_ind_list = range(len(gt_data))
    gt_data['O'] = o_ind_list
    gt_data.to_csv('../../data/flight/cleaned_data/gt_data.csv')


def data():
    data_list = []
    attr_list = ['SD', 'AD', 'DG', 'SA', 'AA', 'AG']
    raw_data = pd.read_csv('../../data/flight/cleaned_data/raw_data.csv', index_col=0).drop_duplicates()
    gt_data = pd.read_csv('../../data/flight/cleaned_data/gt_data.csv', index_col=0)
    for flight in raw_data.iterrows():
        flight = flight[1]
        for attr in attr_list:
            if type(flight[attr]) is not float:
                if flight[attr].strip():
                    if type(flight.SD) is not float:
                        date_f = flight.SD[0:10]
                    elif type(flight.AD) is not float:
                        date_f = flight.AD[0:10]
                    elif type(flight.SA) is not float:
                        date_f = flight.SA[0:10]
                    elif type(flight.AA) is not float:
                        date_f = flight.AA[0:10]
                    obj_name = date_f + '-' + flight.F + '-' + attr
                    data_list.append([flight.S, obj_name, flight[attr]])

    data_f = pd.DataFrame(data=data_list, columns=['S', 'O_name', 'V'])
    data_f.loc[:, 'O'] = None
    gt_o_name_list = gt_data.O_name.values
    for o_name in gt_o_name_list:
        o_ind = gt_data[gt_data.O_name == o_name].O.values[0]
        data_f.at[data_f[data_f.O_name == o_name].index, 'O'] = o_ind
    o_name_rest = list(set(data_f.O_name.drop_duplicates()) - set(gt_o_name_list))
    start_ind = len(gt_o_name_list)
    for o_ind, o_name in enumerate(o_name_rest):
        data_f.at[data_f[data_f.O_name == o_name].index, 'O'] = start_ind + o_ind

    data_f.to_csv('../../data/flight/cleaned_data/data.csv')


def make_buckets():
    gt_data = pd.read_csv('../../data/flight/cleaned_data/gt_data.csv', index_col=0)
    data = pd.read_csv('../../data/flight/cleaned_data/data.csv', index_col=0)
    gt_data_len = len(gt_data.O.values)

    delta = timedelta(minutes=t_delta)
    for ind, obj_ind in enumerate(data.O.drop_duplicates().values):
        val_list = list(data[data.O == obj_ind].V.values)
        try:
            freq_val = pd.to_datetime(max(set(val_list), key=val_list.count))
        except ValueError:
            continue
        buckets_list = [freq_val - timedelta(minutes=t_delta/2.), freq_val + timedelta(minutes=t_delta/2.)]
        possible_values = sorted([pd.to_datetime(i) for i in list(set(val_list))])
        r_n = (max(possible_values).hour*60+max(possible_values).minute -
               buckets_list[1].hour*60-buckets_list[1].minute)/t_delta
        r_bound_number = int(math.ceil(abs(r_n))) if r_n > 0 else 0
        l_n = (buckets_list[0].hour*60+buckets_list[0].minute -
               min(possible_values).hour*60-min(possible_values).minute)/t_delta
        l_bound_number = int(math.ceil(abs(l_n))) if l_n > 0 else 0
        for i in range(r_bound_number):
            buckets_list.append(buckets_list[-1] + delta)
        for j in range(l_bound_number):
            buckets_list.insert(0, buckets_list[0] - delta)

        if obj_ind < gt_data_len:
            gt = gt_data[gt_data.O == obj_ind]
            if len(gt.V.values[0]) == 19:
                gt_val = pd.to_datetime(gt.V.values[0])
                for i in range(len(buckets_list)-1):
                    if (gt_val >= buckets_list[i]) and (gt_val <= buckets_list[i+1]):
                        val_new = gt.O_name + '-' + str(i)
                        gt_data.at[gt.index, 'V'] = val_new

        psi_obj = data[data.O == obj_ind]
        for psi in psi_obj.iterrows():
            psi_ind = psi[0]
            psi = psi[1]
            psi_val = pd.to_datetime(psi.V)
            for i in range(len(buckets_list)-1):
                if (psi_val >= buckets_list[i]) and (psi_val <= buckets_list[i+1]):
                    val_new = psi.O_name + '-' + str(i)
                    data.at[psi_ind, 'V'] = val_new
                    break

    gt_data.to_csv('../../data/flight/cleaned_data/gt_b_data.csv')
    data.to_csv('../../data/flight/cleaned_data/b_data.csv')


if __name__ == '__main__':
    raw_flights()
    # raw_gt()
    # gt()
    # data()
    # make_buckets()
