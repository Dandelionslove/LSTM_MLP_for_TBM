# -*- coding:utf-8 -*-

import pandas as pd
from sklearn.model_selection import train_test_split
import os
from sklearn.preprocessing import StandardScaler
import numpy as np

# model params
TARGET = 'T'  # 当前模型的预测目标；F, T, S
# KFOLD_SPLIT = 5

EPOCHS = 50
# BATCH_SIZE = 32  # 256
# init_lr = 3e-2
decay_rate = 0.9

VALIDATION_SPLIT = 0.2
TEST_STZE = 0.2
kfold_num = 3

# GA params
GEN_LENGTH = 200  # the gen num in one population
NB_GENS = 50  # the generation of GA
k1 = 1.0
k2 = 0.5
k3 = 1.0
k4 = 0.5
retain_value = 0.5
select_value = 0.15
parents_max_ratio = 0.8  # must be larger than retain_value
# breed_children_num = 3

# files
pj_root_path = os.path.abspath(os.path.dirname(__file__))
data_file_dir = str(pj_root_path) + '/Data/'
bp_data_csv = data_file_dir + 'mlp.csv'
lstm_rise_data_csv = data_file_dir + 'rise_time_series.csv'
# normalization_info_file = data_file_dir + TARGET + '_norm_info.txt'

result_file_dir = pj_root_path + '/GAResult/'
if not os.path.exists(result_file_dir):
    os.makedirs(result_file_dir)
bpnn_dir = result_file_dir + '/BPNN/'
if not os.path.exists(bpnn_dir):
    os.makedirs(bpnn_dir)
lstm_dir = result_file_dir + '/LSTM/'
if not os.path.exists(lstm_dir):
    os.makedirs(lstm_dir)
lstm_bpnn_dir = result_file_dir + '/LSTM_BPNN/'
if not os.path.exists(lstm_bpnn_dir):
    os.makedirs(lstm_bpnn_dir)

# ga_result_file_name = 'fittest_model_info_' + TARGET + \
#                  '_NbGens' + str(NB_GENS) + '_GenLength' + str(GEN_LENGTH) + \
#                  '_Epochs' + str(EPOCHS) + '_BatchSize' + str(BATCH_SIZE) + '.txt'
# ga_log_info_file_name = 'log' + TARGET + \
#                    '_NbGens' + str(NB_GENS) + '_GenLength' + str(GEN_LENGTH) + \
#                    '_Epochs' + str(EPOCHS) + '_BatchSize' + str(BATCH_SIZE) + '.txt'
ga_result_file_name = 'fittest_model_info_' + TARGET + \
                 '_NbGens' + str(NB_GENS) + '_GenLength' + str(GEN_LENGTH) + \
                 '_Epochs' + str(EPOCHS) + '.txt'
ga_log_info_file_name = 'log' + TARGET + \
                   '_NbGens' + str(NB_GENS) + '_GenLength' + str(GEN_LENGTH) + \
                   '_Epochs' + str(EPOCHS) + '.txt'
bpnn_ga_result_file = bpnn_dir + ga_result_file_name
bpnn_ga_log_file = bpnn_dir + ga_log_info_file_name
lstm_ga_result_file = lstm_dir + ga_result_file_name
lstm_ga_log_file = lstm_dir + ga_log_info_file_name
lstm_bpnn_ga_result_file = lstm_bpnn_dir + ga_result_file_name
lstm_bpnn_ga_log_file = lstm_bpnn_dir + ga_log_info_file_name

fig_dir = pj_root_path + '/Figures/'
if not os.path.exists(fig_dir):
    os.makedirs(fig_dir)

model_dir = pj_root_path + '/Model/'
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

# fixed variables
F_name = '稳定段总推进力均值'
T_name = '稳定段刀盘扭矩均值'
S_name = '稳定段推进速度均值'
patience_num = 5

time_steps = 22
feature_names = ['上升段推进压力',
                 '上升段总推进力',
                 '上升段撑靴压力',
                 '上升段刀盘功率',
                 '上升段刀盘扭矩',
                 '上升段控制泵压力',
                 '上升段刀盘转速',
                 '上升段贯入度',
                 '上升段推进速度'
                 ]
name1 = [name + '均值' for name in feature_names]
name2 = [name + '方差' for name in feature_names]
bp_feature_names = name1 + name2


# 对原始数据进行处理
def data_preparation(bp_data_csv, lstm_data_csv, target=None):
    bp_data = pd.read_csv(bp_data_csv, sep=',', header=0)

    bp_cols_names = bp_data.columns.values.tolist()

    bp_x_cols_names = bp_feature_names
    bp_x = bp_data[bp_x_cols_names]
    bp_input_length = bp_x.shape[1]
    bp_x_scaler = StandardScaler()
    bp_x_scaler.fit(bp_x)
    bp_x_array = bp_x_scaler.transform(bp_x)

    bp_y_cols_names = bp_cols_names[-3:]
    df_y = bp_data[bp_y_cols_names]
    y_scaler = StandardScaler()
    y_scaler.fit(df_y)
    y_array = y_scaler.transform(df_y)
    df_y = pd.DataFrame(y_array, columns=bp_y_cols_names)
    if target == 'F':
        y_name = F_name
    elif target == 'T':
        y_name = T_name
    else:
        y_name = S_name
    df_y = df_y[y_name]
    y_array = df_y.to_numpy()

    lstm_x_df = pd.read_csv(lstm_data_csv, sep=',', header=0, index_col=0)
    sample_num = int(lstm_x_df.shape[0] / time_steps)
    if sample_num != len(y_array):
        print('data error')
        return None

    lstm_x_cols = feature_names
    lstm_x_df = lstm_x_df[lstm_x_cols]

    lstm_x_values = lstm_x_df.values
    lstm_x_values = lstm_x_values.astype('float32')

    lstm_x_scaler = StandardScaler()
    lstm_x_scaler.fit(lstm_x_values)
    lstm_x_array = lstm_x_scaler.transform(lstm_x_df)
    lstm_x_df = pd.DataFrame(lstm_x_array, columns=lstm_x_cols)
    lstm_input_length = lstm_x_df.shape[1]
    lstm_x_array = lstm_x_df.to_numpy().reshape((sample_num, time_steps, lstm_input_length))

    bp_train_X, bp_test_X, train_y, test_y = train_test_split(bp_x_array, y_array, test_size=TEST_STZE, shuffle=False)
    lstm_train_X, lstm_test_X, train_y, test_y = train_test_split(lstm_x_array, y_array, test_size=TEST_STZE,
                                                                  shuffle=False)

    # print(len(bp_train_X), len(lstm_train_X), len(train_y), len(bp_test_X), len(lstm_test_X), len(test_y))
    # print(lstm_train_X)

    return bp_train_X, bp_test_X, lstm_train_X, lstm_test_X, train_y.reshape(-1), test_y.reshape(
        -1), bp_input_length, lstm_input_length


bp_train_X, bp_test_X, lstm_train_X, lstm_test_X, train_y, test_y, bp_input_length, lstm_input_length = \
    data_preparation(bp_data_csv, lstm_rise_data_csv, target=TARGET)


def compute_pc(f_scores_list, m_score, f_score):
    f_max = f_scores_list[-1]
    f_avg = np.mean(f_scores_list)
    if m_score>f_score:
        f_p = m_score
    else:
        f_p = f_score
    if f_p >= f_avg:
        pc = k1 * (f_max - f_p) / (f_max - f_avg)
    else:
        pc = k3
    return pc


def compute_pm(f_scores_list, f_score):
    f_max = f_scores_list[-1]
    f_avg = np.mean(f_scores_list)
    if f_score >= f_avg:
        pm = k2 * (f_max - f_score) / (f_max - f_avg)
    else:
        pm = k4
    return pm


if __name__ == '__main__':
    data_preparation(bp_data_csv, lstm_rise_data_csv, target='T')
