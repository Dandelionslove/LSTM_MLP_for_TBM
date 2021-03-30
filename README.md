# LSTM_MLP_for_TBM
A sequential combination of LSTM and MLP for the prediction of cutterhead torque for TBM.

Before you run the scripts, please decompress the zip file of the data first.

#### 1. Environment

```txt
python 3.6.5
keras 2.3.1
tensorflow 1.14.0
numpy 1.18.5
pandas 1.0.4
scikit-learn 0.22.1
matplotlib 3.3.4
```

#### 2. Run AGA-optimized LSTM-MLP

```shell
python lstm_bp_gann.py
```

#### 3. Run AGA-optimized LSTMs

```shell
python lstm_gann.py
```

#### 4. Run AGA-optimized BPNN

```shell
python bpnn_gann.py
```

#### 5. Modify Configurations

The parameters of the genetic algorithms can be modified in the file with the suffix of ***ga.py***.

And some other configurations, such as the ration of the test size, the decay rate for learning and so on, which can be modified in file ***VariablesFunctions.py***.