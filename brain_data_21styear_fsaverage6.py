#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import numpy as np
import nibabel as nib
from nibabel.testing import data_path
import matplotlib.pyplot as plt
import pandas as pd


# # Whole Brain Labels

# In[3]:


L_labels = nib.load('../afni-nosmooth/tpl-fsaverage6/tpl-fsaverage6_hemi-L_desc-MMP_dseg.label.gii')
R_labels = nib.load('../afni-nosmooth/tpl-fsaverage6/tpl-fsaverage6_hemi-R_desc-MMP_dseg.label.gii')


# In[15]:


print(L_labels.labeltable.get_labels_as_dict())


# In[16]:


subjects = [244, 249, 254, 255, 256, 257, 258, 259, 260, 261,
           262, 263, 264, 265, 266, 267, 268, 269, 270, 271]


# In[17]:


if not os.path.exists('21styear_braindata'):
    os.mkdir('21styear_braindata')


# In[29]:


count=1
for eachsub in subjects:
    data_voxels_lh = nib.load('../afni-nosmooth/sub-'+str(eachsub)+'/func/sub-'+str(eachsub)+'_task-21styear_space-fsaverage6_hemi-L_desc-clean.func.gii')
    data_voxels_rh = nib.load('../afni-nosmooth/sub-'+str(eachsub)+'/func/sub-'+str(eachsub)+'_task-21styear_space-fsaverage6_hemi-R_desc-clean.func.gii')
    temp = []
    for i in np.arange(np.array(data_voxels_lh.darrays).shape[0]):
        temp.append(data_voxels_lh.darrays[i].data)
    temp1 = []
    for i in np.arange(np.array(data_voxels_rh.darrays).shape[0]):
        temp1.append(data_voxels_rh.darrays[i].data)
    temp = np.concatenate([np.array(temp),np.array(temp1)],axis=1)
    np.save('21styear_braindata/sub_'+str(count), np.array(temp))

