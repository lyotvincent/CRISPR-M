#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np

encoded_dict = {'A': [1, 0, 0, 0], 'T': [0, 1, 0, 0], 'G': [0, 0, 1, 0], 'C': [0, 0, 0, 1], '_': [0, 0, 0, 0], '-': [0, 0, 0, 0]}
pos_dict = {'A':1, 'T':2, 'G':3, 'C':4, '_':5, '-':5}


# In[ ]:


def my_encode_on_off_dim(target_seq, off_target_seq):
    tlen = 24
    target_seq = "-" *(tlen-len(target_seq)) + target_seq
    
    off_target_seq = "-" *(tlen-len(off_target_seq)) + off_target_seq
    target_seq_code = np.array([encoded_dict[base] for base in list(target_seq)])
    off_target_seq_code = np.array([encoded_dict[base] for base in list(off_target_seq)])
    on_off_dim6_codes = []
    for i in range(len(target_seq)):
        diff_code = np.bitwise_or(target_seq_code[i], off_target_seq_code[i])
        dir_code = np.zeros(2)
        if pos_dict[target_seq[i]] == pos_dict[off_target_seq[i]]:
            diff_code = diff_code*-1
            dir_code[0] = 1
            dir_code[1] = 1
        elif pos_dict[target_seq[i]] < pos_dict[off_target_seq[i]]:
            dir_code[0] = 1
        elif pos_dict[target_seq[i]] > pos_dict[off_target_seq[i]]:
            dir_code[1] = 1
        else:
            raise Exception("Invalid seq!", target_seq, off_target_seq)
        on_off_dim6_codes.append(np.concatenate((diff_code, dir_code)))
    on_off_dim6_codes = np.array(on_off_dim6_codes)
    isPAM = np.zeros((24,1))
    isPAM[-3:, :] = 1
    on_off_code = np.concatenate((on_off_dim6_codes, isPAM), axis=1)
    return on_off_code


# In[ ]:




