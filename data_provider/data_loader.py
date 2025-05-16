import os
import pdb

import numpy as np
import pandas as pd
import os
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from lib.utils.timefeatures import time_features
from pathlib import Path ## EEG dataset specific 
import warnings

warnings.filterwarnings('ignore')


class Dataset_ETT_hour(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h'):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        border1s = [0, 12 * 30 * 24 - self.seq_len, 12 * 30 * 24 + 4 * 30 * 24 - self.seq_len]
        border2s = [12 * 30 * 24, 12 * 30 * 24 + 4 * 30 * 24, 12 * 30 * 24 + 8 * 30 * 24]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop(['date'], axis=1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_ETT_minute(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTm1.csv',
                 target='OT', scale=True, timeenc=0, freq='t'):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        border1s = [0, 12 * 30 * 24 * 4 - self.seq_len, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4 - self.seq_len]
        border2s = [12 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 8 * 30 * 24 * 4]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            df_stamp['minute'] = df_stamp.date.apply(lambda row: row.minute, 1)
            df_stamp['minute'] = df_stamp.minute.map(lambda x: x // 15)
            data_stamp = df_stamp.drop(['date'], axis=1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_Custom(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h'):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        '''
        df_raw.columns: ['date', ...(other features), target feature]
        '''
        cols = list(df_raw.columns)
        cols.remove(self.target)
        cols.remove('date')
        df_raw = df_raw[['date'] + cols + [self.target]]
        # print(cols)
        num_train = int(len(df_raw) * 0.7)
        num_test = int(len(df_raw) * 0.2)
        num_vali = len(df_raw) - num_train - num_test
        border1s = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
        border2s = [num_train, num_train + num_vali, len(df_raw)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            # print(self.scaler.mean_)
            # exit()
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop(['date'], axis=1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_Pred(Dataset):
    def __init__(self, root_path, flag='pred', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, inverse=False, timeenc=0, freq='15min', cols=None):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['pred']

        self.features = features
        self.target = target
        self.scale = scale
        self.inverse = inverse
        self.timeenc = timeenc
        self.freq = freq
        self.cols = cols
        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))
        '''
        df_raw.columns: ['date', ...(other features), target feature]
        '''
        if self.cols:
            cols = self.cols.copy()
            cols.remove(self.target)
        else:
            cols = list(df_raw.columns)
            cols.remove(self.target)
            cols.remove('date')
        df_raw = df_raw[['date'] + cols + [self.target]]
        border1 = len(df_raw) - self.seq_len
        border2 = len(df_raw)

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            self.scaler.fit(df_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        tmp_stamp = df_raw[['date']][border1:border2]
        tmp_stamp['date'] = pd.to_datetime(tmp_stamp.date)
        pred_dates = pd.date_range(tmp_stamp.date.values[-1], periods=self.pred_len + 1, freq=self.freq)

        df_stamp = pd.DataFrame(columns=['date'])
        df_stamp.date = list(tmp_stamp.date.values) + list(pred_dates[1:])
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            df_stamp['minute'] = df_stamp.date.apply(lambda row: row.minute, 1)
            df_stamp['minute'] = df_stamp.minute.map(lambda x: x // 15)
            data_stamp = df_stamp.drop(['date'], axis=1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        if self.inverse:
            self.data_y = df_data.values[border1:border2]
        else:
            self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        if self.inverse:
            seq_y = self.data_x[r_begin:r_begin + self.label_len]
        else:
            seq_y = self.data_y[r_begin:r_begin + self.label_len]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_Neuro(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h'):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        train_data = np.load(os.path.join(self.root_path, 'train_data.npy'))
        val_data = np.load(os.path.join(self.root_path, 'val_data.npy'))
        test_data = np.load(os.path.join(self.root_path, 'test_data.npy'))

        train_sensors_last = np.swapaxes(train_data, 1, 2)
        val_sensors_last = np.swapaxes(val_data, 1, 2)
        test_sensors_last = np.swapaxes(test_data, 1, 2)

        train_data_reshaped = train_sensors_last.reshape(-1, train_sensors_last.shape[-1])
        val_data_reshaped = val_sensors_last.reshape(-1, val_sensors_last.shape[-1])
        test_data_reshaped = test_sensors_last.reshape(-1, test_sensors_last.shape[-1])

        if self.scale:
            self.scaler.fit(train_data_reshaped)
            train_data_scaled = self.scaler.transform(train_data_reshaped)
            val_data_scaled = self.scaler.transform(val_data_reshaped)
            test_data_scaled = self.scaler.transform(test_data_reshaped)

        train_scaled_orig_shape = train_data_scaled.reshape(train_sensors_last.shape)
        val_scaled_orig_shape = val_data_scaled.reshape(val_sensors_last.shape)
        test_scaled_orig_shape = test_data_scaled.reshape(test_sensors_last.shape)

        if self.set_type == 0:  # TRAIN
            train_x, train_y = self.make_full_x_y_data(train_scaled_orig_shape)
            self.data_x = train_x
            self.data_y = train_y

        elif self.set_type == 1:  # VAL
            val_x, val_y = self.make_full_x_y_data(val_scaled_orig_shape)
            self.data_x = val_x
            self.data_y = val_y

        elif self.set_type == 2:  # TEST
            test_x, test_y = self.make_full_x_y_data(test_scaled_orig_shape)
            self.data_x = test_x
            self.data_y = test_y

    def make_full_x_y_data(self, array):
        data_x = []
        data_y = []
        for instance in range(0, array.shape[0]):
            for time in range(0, array.shape[1]):
                s_begin = time
                s_end = s_begin + self.seq_len
                r_begin = s_end - self.label_len
                r_end = r_begin + self.label_len + self.pred_len
                if r_end <= array.shape[1]:
                    data_x.append(array[instance, s_begin:s_end, :])
                    data_y.append(array[instance, r_begin:r_end, :])
                else:
                    break
        return data_x, data_y

    def __getitem__(self, index):
        return self.data_x[index], self.data_y[index], 0, 0

    def __len__(self):
        return len(self.data_x)

    def inverse_transform(self, data):
        print('DATLOADER INVERSE_TRANSFORM - this might not do what you want it to anymore')
        return self.scaler.inverse_transform(data)


class Dataset_Earthquake(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h'):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        train_data = np.load(os.path.join(self.root_path, 'train_data.npy'))
        val_data = np.load(os.path.join(self.root_path, 'val_data.npy'))
        test_data = np.load(os.path.join(self.root_path, 'test_data.npy'))

        train_sensors_last = np.swapaxes(train_data, 1, 2)
        val_sensors_last = np.swapaxes(val_data, 1, 2)
        test_sensors_last = np.swapaxes(test_data, 1, 2)

        train_data_reshaped = train_sensors_last.reshape(-1, train_sensors_last.shape[-1])
        val_data_reshaped = val_sensors_last.reshape(-1, val_sensors_last.shape[-1])
        test_data_reshaped = test_sensors_last.reshape(-1, test_sensors_last.shape[-1])

        if self.scale:
            self.scaler.fit(train_data_reshaped)
            train_data_scaled = self.scaler.transform(train_data_reshaped)
            val_data_scaled = self.scaler.transform(val_data_reshaped)
            test_data_scaled = self.scaler.transform(test_data_reshaped)

        train_scaled_orig_shape = train_data_scaled.reshape(train_sensors_last.shape)
        val_scaled_orig_shape = val_data_scaled.reshape(val_sensors_last.shape)
        test_scaled_orig_shape = test_data_scaled.reshape(test_sensors_last.shape)

        if self.set_type == 0:  # TRAIN
            train_x, train_y = self.make_full_x_y_data(train_scaled_orig_shape)
            self.data_x = train_x
            self.data_y = train_y

        elif self.set_type == 1:  # VAL
            val_x, val_y = self.make_full_x_y_data(val_scaled_orig_shape)
            self.data_x = val_x
            self.data_y = val_y

        elif self.set_type == 2:  # TEST
            test_x, test_y = self.make_full_x_y_data(test_scaled_orig_shape)
            self.data_x = test_x
            self.data_y = test_y

        print(self.set_type, len(self.data_x), len(self.data_y), self.data_x[0].shape, self.data_y[0].shape)

    def make_full_x_y_data(self, array):
        data_x = []
        data_y = []
        for instance in range(0, array.shape[0]):
            for time in range(0, array.shape[1]):
                s_begin = time
                s_end = s_begin + self.seq_len
                r_begin = s_end - self.label_len
                r_end = r_begin + self.label_len + self.pred_len
                if r_end <= array.shape[1]:
                    data_x.append(array[instance, s_begin:s_end, :])
                    data_y.append(array[instance, r_begin:r_end, :])
                else:
                    break
        return data_x, data_y

    def __getitem__(self, index):
        return self.data_x[index], self.data_y[index], 0, 0

    def __len__(self):
        return len(self.data_x)

    def inverse_transform(self, data):
        print('DATLOADER INVERSE_TRANSFORM - this might not do what you want it to anymore')
        return self.scaler.inverse_transform(data)
    
class Dataset_EEG(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features=None, data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h'):
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.step_size = 1
        if (self.target == 'OT') or (self.target.lower() == 'half'):
            self.step_size = self.seq_len // 2
        elif self.target == "1": 
            self.step_size = 1
        else: 
            self.step_size = int(self.target)
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        # self.event_dict = {'Up': 1, 'Down': 2, 'Left': 3, 'Right': 4, 'Rest': 6}
        # Edit this line in data_loader.py
        self.event_dict = {'Rest': 0, 'Event1': 1, 'Event2': 2}

        self.root_path = root_path
        self.data_path = data_path
        self.data_name = Path(data_path).stem 

        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                            self.data_path), index_col=0)
        df_split = pd.read_csv(os.path.join(self.root_path, 
                                            self.data_name + "-split.csv"), index_col=0)
        '''
        df_raw.columns: ['date', ...(other features), target feature]
        '''

        self.eeg_columns = df_raw.columns[:128]

        if self.features == 'clean_data':
            # # Remove columns with indices 2 and 4 from self.eeg_columns
            channels_to_remove = np.genfromtxt(os.path.join(self.root_path, 
                                            self.data_name + "-bad_channels.csv"), delimiter=',')
            filtered_columns = [col for idx, col in enumerate(self.eeg_columns) if idx not in channels_to_remove]

            # Update self.eeg_columns with the filtered columns
            self.eeg_columns = filtered_columns

        
        if self.scale:
            self.scaler.fit(self.make_contiguous_x_data(df_raw, df_split, split='train')) 
            all_x_scaled = self.scaler.transform(df_raw.loc[:, self.eeg_columns])
            df_raw.loc[:, self.eeg_columns] = all_x_scaled

        if self.set_type == 0:  # TRAIN
            train_x, train_y, train_x_label, train_y_label = self.make_full_x_y_data(df_raw, df_split, split='train')
            self.data_x = train_x
            self.data_y = train_y
            self.data_x_label = train_x_label
            self.data_y_label = train_y_label

        elif self.set_type == 1:  # VAL
            val_x, val_y, val_x_label, val_y_label = self.make_full_x_y_data(df_raw, df_split, split='val')
            self.data_x = val_x
            self.data_y = val_y
            self.data_x_label = val_x_label
            self.data_y_label = val_y_label

        elif self.set_type == 2:  # TEST
            test_x, test_y, test_x_label, test_y_label = self.make_full_x_y_data(df_raw, df_split, split='test')
            self.data_x = test_x
            self.data_y = test_y
            self.data_x_label = test_x_label
            self.data_y_label = test_y_label

    """
    Instead of considering each trial ends with a Rest event, we consider the last trial to end with the last event
    (which is not Rest). 
    Redo functions make_contiguous_x_data and make_full_x_y_data to consider this.
    
    

    def make_contiguous_x_data(self, df_raw, df_split, split):         
        data_x = []
        for trial_start_ind, r in df_split[df_split['split'] == split].iterrows():
            trial_end_ind = df_raw.loc[trial_start_ind:][df_raw.loc[trial_start_ind:, 'STI'] == self.event_dict['Rest']].iloc[0].name
            data_x.append(df_raw.loc[trial_start_ind:trial_end_ind, self.eeg_columns].values)
        return np.concatenate(data_x)
            
    def make_full_x_y_data(self, df_raw, df_split, split):
        data_x = []
        data_y = []
        data_x_label = []
        data_y_label = []
        counter = 0
        for trial_start_ind, r in df_split[df_split['split'] == split].iterrows():
            trial_end_ind = df_raw.loc[trial_start_ind:][df_raw.loc[trial_start_ind:, 'STI'] == self.event_dict['Rest']].iloc[0].name
            for time in range(trial_start_ind, trial_end_ind, self.step_size):
                counter += 1
                s_begin = time
                s_end = s_begin + self.seq_len
                r_begin = s_end - self.label_len
                r_end = r_begin + self.label_len + self.pred_len
                if r_end <= trial_end_ind:
                    data_x.append(df_raw.loc[s_begin:s_end-1, self.eeg_columns].values)
                    data_y.append(df_raw.loc[r_begin:r_end-1, self.eeg_columns].values)
                    data_x_label.append(int(r['STI']) - 1)
                    data_y_label.append(int(r['STI']) - 1) ## These labels are 1 indexed in the original files
                else:
                    break
        return data_x, data_y, data_x_label, data_y_label
    """
    def make_contiguous_x_data(self, df_raw, df_split, split):
        data_x = []
        # Get all trial start indices for this split
        trial_starts = df_split[df_split['split'] == split].index.tolist()
    
        # Sort them to ensure they're in ascending order
        trial_starts.sort()
    
        for i, trial_start_ind in enumerate(trial_starts):
            # If this is not the last trial, use the next trial start as the end
            if i < len(trial_starts) - 1:
                trial_end_ind = trial_starts[i+1] - 1  # End just before next trial
            else:
                # For the last trial, go to the end of the dataset
                trial_end_ind = df_raw.index[-1]
            
            data_x.append(df_raw.loc[trial_start_ind:trial_end_ind, self.eeg_columns].values)
        
        if not data_x:  # If no data was collected, return empty array with correct columns
            return np.zeros((0, len(self.eeg_columns)))
        return np.concatenate(data_x)

    def make_full_x_y_data(self, df_raw, df_split, split):
        data_x = []
        data_y = []
        data_x_label = []
        data_y_label = []
    
        # Get all trial start indices for this split
        trial_starts = df_split[df_split['split'] == split].index.tolist()
        trial_starts.sort()
    
        for i, trial_start_ind in enumerate(trial_starts):
            r = df_split.loc[trial_start_ind]
        
            # If this is not the last trial, use the next trial start as the end
            if i < len(trial_starts) - 1:
                trial_end_ind = trial_starts[i+1] - 1
            else:
                # For the last trial, go to the end of the dataset
                trial_end_ind = df_raw.index[-1]
            
            for time in range(trial_start_ind, trial_end_ind, self.step_size):
                s_begin = time
                s_end = s_begin + self.seq_len
                r_begin = s_end - self.label_len
                r_end = r_begin + self.label_len + self.pred_len
            
                if r_end <= trial_end_ind:
                    data_x.append(df_raw.loc[s_begin:s_end-1, self.eeg_columns].values)
                    data_y.append(df_raw.loc[r_begin:r_end-1, self.eeg_columns].values)
                    data_x_label.append(int(r['STI']))  # Keep original STI value (no -1)
                    data_y_label.append(int(r['STI']))  # Keep original STI value
                else:
                    break
                
        return data_x, data_y, data_x_label, data_y_label
    
    def __getitem__(self, index):
        return self.data_x[index], self.data_y[index], self.data_x_label[index], self.data_y_label[index]

    def __len__(self):
        return len(self.data_x)

    def inverse_transform(self, data):
        print('DATLOADER INVERSE_TRANSFORM - this might not do what you want it to anymore')
        return self.scaler.inverse_transform(data)