# -*- coding:utf-8 -*-

from keras import backend
from tensorflow import keras
from VariablesFunctions import *
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold, cross_validate
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
from sklearn import preprocessing

import warnings

warnings.filterwarnings("ignore")
# BATCH_SIZE = 32


class LSTM_NN:
    def __init__(self, nn_parameters=None):

        self.nn_parameters = nn_parameters
        # self.bp_hidden_layer_num = nn_parameters['bp_hidden_layer_num']
        # self.bp_hidden_layer_dims = nn_parameters['bp_hidden_layer_dims']
        self.lstm_units = nn_parameters['units']
        self.optimizer = nn_parameters['optimizer']
        self.dense_dim = nn_parameters['layer_dims']
        self.layer_num = nn_parameters['layer_num']
        self.activation = nn_parameters['activation']
        self.lr = nn_parameters['lr']
        self.batch_size = nn_parameters['batch_size']
        self.bp_input_length = bp_input_length
        self.lstm_input_length = lstm_input_length

    def build_model(self):

        lstm_input_x = keras.layers.Input(shape=(time_steps, self.lstm_input_length))
        # lstm_layer_x = keras.layers.Bidirectional(keras.layers.LSTM(self.lstm_units, activation='tanh'))(lstm_input_x)
        lstm_layer_x = keras.layers.LSTM(
            self.lstm_units,
            dropout=0.1,
            # kernel_regularizer=keras.regularizers.l2(0.2)
        )(lstm_input_x)
        for i in range(self.layer_num):
            lstm_layer_x = keras.layers.Dense(
                self.dense_dim[i],
                activation=self.activation,
                kernel_initializer='random_normal',
                use_bias=True,
                bias_initializer=keras.initializers.Constant(0.1)
            )(lstm_layer_x)
        lstm_output_x = keras.layers.Dense(1)(lstm_layer_x)

        model = keras.models.Model(inputs=lstm_input_x, outputs=lstm_output_x)
        model.summary()
        model.compile(optimizer=self.get_optimizer(), loss='mse', metrics=['mse', 'mae'])
        return model

    def train(self):
        model = self.build_model()
        callback = keras.callbacks.EarlyStopping(monitor='loss', patience=patience_num, restore_best_weights=True)
        history = model.fit([lstm_train_X, bp_train_X], train_y,
                            epochs=EPOCHS,
                            batch_size=self.batch_size,
                            validation_split=VALIDATION_SPLIT,
                            verbose=1,
                            callbacks=[callback])
        pred_y = model.predict([lstm_test_X, bp_test_X], verbose=0)
        test_r2 = r2_score(test_y, pred_y)
        test_mse = mean_squared_error(test_y, pred_y)
        test_mae = mean_absolute_error(test_y, pred_y)
        print('test r2:', test_r2)
        print('test mse:', test_mse)
        print('test mae:', test_mae)
        # self.visualization(history, test_y, pred_y)
        # with open('Results/lstm.txt', 'w') as f:
        #     f.write('test r2:' + str(test_r2) + '\n')
        #     f.write('test mse:' + str(test_mse) + '\n')
        #     f.write('test mae:' + str(test_mae) + '\n')
        #     f.close()

        return test_r2

    def k_train(self):
        estimator = keras.wrappers.scikit_learn.KerasRegressor(build_fn=self.build_model,
                                                               epochs=EPOCHS,
                                                               batch_size=self.batch_size,
                                                               verbose=1)
        scores = cross_validate(estimator, lstm_train_X, train_y, cv=kfold_num,
                                scoring=['r2', 'neg_mean_squared_error', 'neg_mean_absolute_error'],
                                return_train_score=True
                                )
        val_r2 = np.mean(scores['test_r2'])
        val_mse = np.mean(scores['test_neg_mean_squared_error'])
        val_mae = np.mean(scores['test_neg_mean_absolute_error'])
        print('val r2:', scores['test_r2'], val_r2)
        print('val neg mse', scores['test_neg_mean_squared_error'], val_mse)
        print('val neg mae', scores['test_neg_mean_absolute_error'], val_mae)
        cross_train_r2 = np.mean(scores['train_r2'])
        cross_train_mse = np.mean(np.abs(scores['train_neg_mean_squared_error']))
        cross_train_mae = np.mean(np.abs(scores['train_neg_mean_absolute_error']))

        cross_val_r2 = val_r2
        cross_val_mse = np.mean(np.abs(scores['test_neg_mean_squared_error']))
        cross_val_mae = np.mean(np.abs(scores['test_neg_mean_absolute_error']))

        callback = keras.callbacks.EarlyStopping(monitor='loss', patience=5, restore_best_weights=True)
        history = estimator.fit(lstm_train_X, train_y,
                                epochs=30,
                                batch_size=self.batch_size,
                                # validation_split=VALIDATION_SPLIT,
                                verbose=1,
                                callbacks=[callback])
        pred_y = estimator.model.predict(lstm_test_X, verbose=0)
        test_r2 = r2_score(test_y, pred_y)
        test_mse = mean_squared_error(test_y, pred_y)
        test_mae = mean_absolute_error(test_y, pred_y)

        with open('Results/lstm_kfold.txt', 'w') as f:
            f.write('test r2:' + str(test_r2) + '\n')
            f.write('test mse:' + str(test_mse) + '\n')
            f.write('test mae:' + str(test_mae) + '\n')
            f.close()

        self.visualization(history, test_y, pred_y, is_k_fold=True)

    def get_optimizer(self):
        if self.optimizer == 'Adagrad':
            opt = keras.optimizers.Adagrad(learning_rate=self.lr, decay=decay_rate)
        elif self.optimizer == 'Adadelta':
            opt = keras.optimizers.Adadelta(learning_rate=self.lr, decay=decay_rate)
        elif self.optimizer == 'sgd':
            opt = keras.optimizers.SGD(learning_rate=self.lr, decay=decay_rate)
        else:
            opt = keras.optimizers.Adam(learning_rate=self.lr, decay=decay_rate)
        return opt

    def visualization(self, history, test_y, pred_y, is_k_fold=False):
        loss_fig = fig_dir + TARGET + '_lstmnn_loss.png'
        diff_fig = fig_dir + TARGET + '_lstmnn_loss_diff.png'
        fitting_fig = fig_dir + TARGET + '_lstmnn_fitting_result.png'
        fitting_fig_2 = fig_dir + TARGET + '_lstmnn_fitting_result_2.png'

        # plot loss
        if not is_k_fold:
            train_loss = history.history['loss']
            val_loss = history.history['val_loss']
            plt.figure()
            plt.plot(train_loss, color='red', label='training loss')
            plt.plot(val_loss, color='blue', label='validation loss')
            plt.legend(loc='upper right')
            plt.title('loss')
        else:
            train_loss = history.history['loss']
            plt.figure()
            plt.plot(train_loss, color='red')
            plt.title('training loss')
        plt.savefig(loss_fig)
        plt.show()

        length = len(test_y)
        if length > 150:
            length = 150

        # # plot diff
        # plt.figure()
        # diff = np.abs(test_y - pred_y).reshape(-1, 1)
        # plt.bar(range(length), diff, fc='gray')
        # plt.title('The difference between the true values and the predicted values')
        # plt.savefig(diff_fig)
        # plt.show()

        # plot fitting1
        plt.figure()
        plt.plot(range(length), test_y[:length], 'mediumturquoise')
        plt.plot(range(length), pred_y[:length], 'lightcoral')
        plt.plot(range(length), test_y[:length], 'o', markersize=1, c='darkgreen', label='true')
        plt.plot(range(length), pred_y[:length], '^', markersize=1, c='brown', label='prediction')
        plt.title("fitting result on test data")
        plt.legend(loc='upper right')
        plt.savefig(fitting_fig)
        plt.show()

        # # plot fitting2
        # plt.figure()
        # plt.plot(test_y, test_y, 'mediumturquoise', label='true')
        # plt.scatter(test_y, pred_y, s=4, c='lightcoral', alpha=0.8, label='prediction')
        # plt.title('fitting result on test data')
        # plt.legend(loc='upper right')
        # plt.savefig(fitting_fig_2)
        # plt.show()


if __name__ == '__main__':
    # nn_parameters = {
    #     'lstm_units': 64,
    #     'optimizer': 'Adam',
    #     'dense_dim': [32, 8]
    # }
    # # bp_train_X, bp_test_X, lstm_train_X, lstm_test_X, train_y, test_y, bp_input_length, lstm_input_length =\
    # #     data_preparation(bp_data_csv, lstm_rise_data_csv, target=TARGET)
    # lstm_nn = LSTM_NN(nn_parameters=nn_parameters)
    # lstm_nn.train()
    # # lstm_nn.k_train()
    print([i for i in range(0)])
