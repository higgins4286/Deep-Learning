import numpy as np
import pandas as pd

'''
Part A

Step 1. Find the inputs that have the highest cross correlation coefficient.
'''

df_fat = pd.read_csv('~/Downloads/bodyfat.csv')

X = df_fat.loc[:, 'Age':'Wrist']
Targ = df_fat.loc[:, 'BodyFat']

from sklearn.feature_selection import r_regression

Pierce = []
Pierce = r_regression(X, Targ)
col_names = np.array(X.columns.tolist())

dat = {'Cross_Coeff': Pierce, 'Inputs': col_names}

lin_reg_df = pd.DataFrame(data=dat)
lin_reg_ordered = lin_reg_df.sort_values(by='Cross_Coeff', ascending=False)

'''
Step 2. Use the inputs that are highly linearly correlated to find the multi-
        variate liner regression. Split data into 50/50 testing/training sets
        and find the MSE.
        
Note: Since a high degree of correlation is between the range of 0.5 to 1, 
      I choose:
          Abdomen, Chest, Hip, Weight, Thigh, and Knee
'''

# Create the highly correlated dataset

high_r_cols = lin_reg_ordered['Inputs'][0:6].tolist()
data = [df_fat[name] for name in high_r_cols]
X_high_r = pd.DataFrame(data).T

# Split into testing/training based on index

X_train = X_high_r[0:126]
Targ_train = Targ[0:126]

X_test =  X_high_r[126:252]
Targ_test = Targ[126:252]

# Multivariate Linear Regression

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

Lin_Reg = LinearRegression()
Lin_Reg.fit(X_train, Targ_train)

Targ_pred = Lin_Reg.predict(X_test)

mean_err_test = mean_squared_error(Targ_test, Targ_pred)
mean_err_train = mean_squared_error(Targ_train, Targ_pred)

## Narrowing the correlated values to Abdomen, Chest, Hip, and Weight

X_higher_r = X_high_r.loc[:, 'Abdomen':'Weight']

# Split into testing/training based on index

X_train2 = X_higher_r[0:126]
Targ_train2 = Targ[0:126]

X_test2 =  X_higher_r[126:252]
Targ_test2 = Targ[126:252]

# Multivariate Linear Regression

Lin_Reg2 = LinearRegression()
Lin_Reg2.fit(X_train2, Targ_train2)

Targ_pred2 = Lin_Reg2.predict(X_test2)

mean_err_test2 = mean_squared_error(Targ_test2, Targ_pred2)
mean_err_train2 = mean_squared_error(Targ_train2, Targ_pred2)

print(mean_err_test2, mean_err_train2)

'''
Part B

Use different MLPs on all features.

For each requested task, reset the network and retrain it 10 times. Report on 
both mean and variance on training and validation MSEs.

Note: The default for sklearn MLPRegressor and train_test_split is random 
      initialization.
'''
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split

'''
Network #1

1. Create a simple 10-node, one hidden layer regression MLP.
2. Use the network with its default settings except for training/validation/test data
   partitioning ratios.
3. Set to Training to 80%, Validation to 20%, and Testing to 0 %
4. Use random initializations

'''
# Define network
MLP_1 = MLPRegressor(hidden_layer_sizes=(10,))

# Assign train and validation sets
X_trainN1, X_valN1, T_trainN1, T_valN1 = train_test_split(X, Targ, 
                                                          test_size=0.2, 
                                                          train_size=0.8)

# Train & test network

epochs = np.arange(1, 11)

mean_N1_train, mean_N1_val, var_N1_train, var_N1_val = [], [], [], []

for epoch in range(epochs.min(), epochs.max()+1):

    MLP_1.fit(X_trainN1, T_trainN1)
    T_predN1 = MLP_1.predict(X_trainN1)
    
    mN1_train = mean_squared_error(T_trainN1, T_predN1)
    mN1_val = mean_squared_error(T_valN1, MLP_1.predict(X_valN1))
    
    vN1_tr = np.var(T_predN1)
    vN1_va = np.var(MLP_1.predict(X_valN1))
    
    mean_N1_train.append(mN1_train)
    mean_N1_val.append(mN1_val)
    var_N1_train.append(vN1_tr)
    var_N1_val.append(vN1_va)

import statistics as stat

# Save outputs to dataframe
N1_dat = [(mean_N1_train[i-1], mean_N1_val[i-1], 
           var_N1_train[i-1], var_N1_val[i-1]) for i in range(epochs.min(), epochs.max()+1)]
N1_scores = pd.DataFrame(N1_dat, 
                       columns=['Train MSE', 'Val MSE', 'Train Var', 'Val Var']).set_index(epochs)

# Calculate score averages across each epoch and append to 
# N1_scores dataframe for ease of model compairson.
N1_score_avg = [stat.mean(mean_N1_train), stat.mean(mean_N1_val), 
                stat.mean(var_N1_train), stat.mean(var_N1_val)]

N1_scores.loc['Score Avgs'] = N1_score_avg

'''
Network #2

1. Change network size to 2 nodes
2. Set to Training to 30%, Validation to 70%, and Testing to 0 %
3. Use random initializations
'''

MLP_2 = MLPRegressor(hidden_layer_sizes=(2,) )
X_trainN2, X_valN2, T_trainN2, T_valN2 = train_test_split(X, Targ, 
                                                          test_size=0.7, 
                                                          train_size=0.3)

mean_N2_train, mean_N2_val, var_N2_train, var_N2_val = [], [], [], []

for epoch in range(epochs.min(), epochs.max()+1):

    MLP_2.fit(X_trainN2, T_trainN2)
    T_predN2 = MLP_2.predict(X_trainN2)
    
    mN2_train = mean_squared_error(T_trainN2, T_predN2)
    mN2_val = mean_squared_error(T_valN2, MLP_2.predict(X_valN2))
    
    vN2_tr = np.var(T_predN2)
    vN2_va = np.var(MLP_2.predict(X_valN2))
    
    mean_N2_train.append(mN2_train)
    mean_N2_val.append(mN2_val)
    var_N2_train.append(vN2_tr)
    var_N2_val.append(vN2_va)

import pandas as pd

N2_dat = [(mean_N2_train[i-1], mean_N2_val[i-1], 
           var_N2_train[i-1], var_N2_val[i-1]) for i in range(epochs.min(), epochs.max()+1)]
N2_scores = pd.DataFrame(N2_dat, 
                       columns=['Train MSE', 'Val MSE', 'Train Var', 'Val Var']).set_index(epochs)

N2_score_avg = [stat.mean(mean_N2_train), stat.mean(mean_N2_val), 
                stat.mean(var_N2_train), stat.mean(var_N2_val)]

N2_scores.loc['Score Avgs'] = N2_score_avg

'''
Network #3

1. 50 nodes
2. Set to Training to 30%, Validation to 70%, and Testing to 0 %
3. Set regularization (weight decay) to 0.1
4. Use random initializations
'''

MLP_3 = MLPRegressor(hidden_layer_sizes=(50,), alpha=0.1)
X_trainN3, X_valN3, T_trainN3, T_valN3 = train_test_split(X, Targ, 
                                                          test_size=0.7, 
                                                          train_size=0.3)

mean_N3_train, mean_N3_val, var_N3_train, var_N3_val = [], [], [], []

for epoch in range(epochs.min(), epochs.max()+1):

    MLP_3.fit(X_trainN3, T_trainN3)
    T_predN3 = MLP_3.predict(X_trainN3)
    
    mN3_train = mean_squared_error(T_trainN3, T_predN3)
    mN3_val = mean_squared_error(T_valN3, MLP_3.predict(X_valN3))
    
    vN3_tr = np.var(T_predN3)
    vN3_va = np.var(MLP_3.predict(X_valN3))
    
    mean_N3_train.append(mN3_train)
    mean_N3_val.append(mN3_val)
    var_N3_train.append(vN3_tr)
    var_N3_val.append(vN3_va)

N3_dat = [(mean_N3_train[i-1], mean_N3_val[i-1], 
           var_N3_train[i-1], var_N3_val[i-1]) for i in range(epochs.min(), epochs.max()+1)]
N3_scores = pd.DataFrame(N3_dat, 
                       columns=['Train MSE', 'Val MSE', 'Train Var', 'Val Var']).set_index(epochs)

N3_score_avg = [stat.mean(mean_N3_train), stat.mean(mean_N3_val), 
                stat.mean(var_N3_train), stat.mean(var_N3_val)]

N3_scores.loc['Score Avgs'] = N3_score_avg
'''
Network #4

1. 50 nodes
2. Set to Training to 30%, Validation to 70%, and Testing to 0 %
3. Set regularization (weight decay) to 0.5
4. Use random initializations
'''

MLP_4 = MLPRegressor(hidden_layer_sizes=(50,), alpha=0.5 )
X_trainN4, X_valN4, T_trainN4, T_valN4 = train_test_split(X, Targ, 
                                                          test_size=0.7, 
                                                          train_size=0.3)

mean_N4_train, mean_N4_val, var_N4_train, var_N4_val = [], [], [], []

for epoch in range(epochs.min(), epochs.max()+1):

    MLP_4.fit(X_trainN4, T_trainN4)
    T_predN4 = MLP_4.predict(X_trainN4)
    
    mN4_train = mean_squared_error(T_trainN4, T_predN4)
    mN4_val = mean_squared_error(T_valN4, MLP_4.predict(X_valN4))
    
    vN4_tr = np.var(T_predN4)
    vN4_va = np.var(MLP_4.predict(X_valN4))
    
    mean_N4_train.append(mN4_train)
    mean_N4_val.append(mN4_val)
    var_N4_train.append(vN4_tr)
    var_N4_val.append(vN4_va)

N4_dat = [(mean_N4_train[i-1], mean_N4_val[i-1], 
           var_N4_train[i-1], var_N4_val[i-1]) for i in range(epochs.min(), epochs.max()+1)]
N4_scores = pd.DataFrame(N4_dat, 
                       columns=['Train MSE', 'Val MSE', 
                                'Train Var', 'Val Var']).set_index(epochs)

N4_score_avg = [stat.mean(mean_N4_train), stat.mean(mean_N4_val), 
                stat.mean(var_N4_train), stat.mean(var_N4_val)]

N4_scores.loc['Score Avgs'] = N4_score_avg


# Print all 4 network results to take screenshots and copy into slide stack.
print('\n', N1_scores, '\n', '\n', N2_scores, '\n', '\n', N3_scores, 
      '\n', '\n', N4_scores, '\n')



















