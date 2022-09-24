#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from encoding import my_encode_on_off_dim
import CRISPR_IP
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical
import os
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, roc_auc_score, average_precision_score


# In[2]:


seed = 123
os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1'
os.environ['PYTHONHASHSEED']=str(seed)
np.random.seed(seed)
tf.random.set_seed(seed)


# In[3]:


num_classes = 2
encoder_shape=(24,7)
seq_len, coding_dim = encoder_shape


# In[4]:


print('Load model!!')
model = load_model('./example+crispr_ip.h5')
print('Encoding!!')
test_data = pd.read_csv('./example-test-data.csv')
test_data_encodings = np.array(test_data.apply(lambda row: my_encode_on_off_dim(row['sgRNAs'], row['DNAs']), axis = 1).to_list())
test_labels = test_data.loc[:, 'labels'].values
print('End of the encoding!!')


# In[5]:


input_shape = (1, seq_len, coding_dim)
xtest = test_data_encodings.reshape(test_data_encodings.shape[0], 1, seq_len, coding_dim)
xtest = xtest.astype('float32')
ytest = test_labels


# In[6]:


yscore = model.predict(xtest)
print(yscore)
for i, y in enumerate(yscore):
    if y[1] > y[0]:
        print(i)
ypred = np.argmax(yscore, axis=1)
for i, y in enumerate(ypred):
    if y > 0:
        print(i)
print(ypred)
yscore = yscore[:,1]


# In[7]:


test_data['pred_label'] = ypred
test_data['pred_score'] = yscore


# In[8]:


# test_data.to_csv('example_saved/example-predict-result.csv', index=False)
# print('Saved result!!')


# In[9]:


eval_funs = [accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, average_precision_score]
eval_fun_names = ['Accuracy', 'F1 score', 'Precision', 'Recall', 'ROC AUC', 'PR AUC']
eval_fun_types = [True, True, True, True, False, False]


# In[10]:


for index_f, function in enumerate(eval_funs):
    if eval_fun_types[index_f]:
        score = np.round(function(ytest, ypred), 4)
    else:
        score = np.round(function(ytest, yscore), 4)
    print('{:<15}{:>15}'.format(eval_fun_names[index_f], score))


# In[ ]:




