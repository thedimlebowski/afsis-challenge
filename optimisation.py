#!/usr/bin/python
# coding: utf-8

# In[2]:

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import random as rand

from NeuralNetworkOptimisation import NeuralNetworkOptimisation
from PCA import PCA


# In[3]:

train = pd.read_csv('training.csv')
train = train.replace({'Topsoil' : 1, 'Subsoil' : -1})
test = pd.read_csv('sorted_test.csv')
test = test.replace({'Topsoil' : 1, 'Subsoil' : -1})


# In[4]:

wavelength_indices = list(np.arange(1,3579))[::1]
param_indices = list(np.arange(3579,3595))
outputs_indices = list(np.arange(3595,3600))


# In[5]:

train_param_df = train.iloc[:,param_indices]
train_spectra_df = train.iloc[:,wavelength_indices]
train_spectra_array = train_spectra_df.values


# In[6]:

#pca = PCA(30)
#print np.shape(train_spectra_array)
#pca.fit(train_spectra_array)
#train_spectra_array = pca.predict(train_spectra_array)
#print np.shape(train_spectra_array)


# In[7]:

train_spectra_df = pd.DataFrame(train_spectra_array)
train_in_df = pd.concat([train_spectra_df,train_param_df], axis =1)
train_in = train_in_df.values
train_out = train.iloc[:,outputs_indices].values


# In[8]:

test_param_df = test.iloc[:,param_indices]
test_spectra_df = test.iloc[:,wavelength_indices]
test_spectra_array = test_spectra_df.values


# In[9]:

#test_spectra_array = pca.predict(test_spectra_array)


# In[10]:

test_spectra_df = pd.DataFrame(test_spectra_array)
test_in_df = pd.concat([test_spectra_df,test_param_df], axis =1)
test_in = test_in_df.values


# In[15]:

n_sample = np.shape(train_in)[0]
n_input = np.shape(train_in)[1]
n_output = np.shape(train_out)[1]

net_arch = (n_input, n_input,  n_output)
print net_arch


# In[16]:

nno = NeuralNetworkOptimisation(net_arch, verbose = 1)


# In[ ]:

#nno.train_cv(train_in, train_out, nproc = 4, ratio = 0.0, best_test = False, ntry = 20, ncv = 20)
nno.train_best(train_in, train_out, nproc = 4, ntry = 20)

# In[62]:

test_out = nno.sim(test_in, cv = True)
print np.shape(test_in),np.shape(test_out), np.shape(train_out)
test_out_df = pd.concat([test[['PIDN']],pd.DataFrame(test_out, columns = ['Ca', 'P', 'pH', 'SOC', 'Sand'])], axis = 1)
test_out_df.to_csv('submission.csv', index = 0)
test_out_df

