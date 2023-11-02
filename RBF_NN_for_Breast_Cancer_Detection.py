"""
Project 3: RBF NN for Breast Cancer Detection, and Deep NN Intro
"""

'''
Part A: RBF Classification, 60 points

Done in Python.
'''

# Open .mat files with scipy.io and assign P and T labels
from scipy.io import loadmat


#Load datasets.
P_dict = loadmat('/Users/laurenhiggins/Dropbox/Machine_Learning/Projects/BCdata/P.mat', 
            mat_dtype=True)
T_dict = loadmat('/Users/laurenhiggins/Dropbox/Machine_Learning/Projects/BCdata/T.mat', 
            mat_dtype=True)

#Convert dictionaries into numpy arrays for use in train_test_split below.
P = P_dict['P'].T
T = T_dict['T'].T

### Create the training (60%), validation(20%), and testing(20%) sets
from sklearn.model_selection import train_test_split

# Split data into 60% train and 40% intermediate set to use in next step
P_train, P_mid, T_train, T_mid = train_test_split(P, T, test_size=0.4, 
                                                        train_size=0.6, random_state=0)

# Split the intermediate 40% of data into a 50/50 split to create the 
# 20% test and 20% validation sets
P_test, P_val, T_test, T_val = train_test_split(P_mid, T_mid, test_size=0.5, 
                                                              train_size=0.5, random_state=0)





'''
Part B: Deep NN Intro, 40 points

Done in MatLab. See screenshot of results in slide deck.
'''


