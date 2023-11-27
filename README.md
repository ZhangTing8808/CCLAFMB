# CCLAFMB

This repository contains TensorFlow codes and datasets for the paper:

> Jianxing Zheng, Ting Zhang, Suge Wang, Deyu Lia∗, Jian Liao


## Introduction
Combined contrast learning for recommendation with adaptive fusion of multi-behavior

## Citation 
```
@inproceedings{chen2021graph,
  title={Graph Heterogeneous Multi-Relational Recommendation},
  author={Chen, Chong and Ma, Weizhi and Zhang, Min and Wang, Zhaowei and He, Xiuqiang and Wang, Chenyang and Liu, Yiqun and Ma, Shaoping},
  booktitle={Proceedings of AAAI},
  year={2021},
}
```

## Environment
The codes of CCLAFMB are implemented and tested under the following development environment:
* python=3.7.16
* tensorflow=1.14.0
* numpy=1.21.6
* scipy=1.6.2


## Dataset
We utilized two datasets to evaluate CCLAFMB: Beibei and Taobao. For both datasets,the Purchase behavior is taken as the target behavior. The last target behavior for the test users are left out to compose the testing set. For both datasets, we remove some user and item records that are less than 5 purchasing behavior due to the sparsity. 


## Example to run the codes		
Train and evaluate our model:
```
python CCLAFMB.py
```
# Beibei
```
python CCLAFMB.py --dataset=Beibei --embed_size=128 --layer_size=[128,128,128] --batch_size=256 --lr=0.001 --wid=[0.1,0.1,0.1] --decay=10 --decay_cl=100 --coefficient=[0.05,0.80,0.15] --mess_dropout=0.2 --node_dropout=0.1 
```
# taobao 
```
python CCLAFMB.py --dataset=Taobao --embed_size=256 --layer_size=[256,256,256] --batch_size=256 --lr=0.001 --wid=[0.01,0.01,0.01] --decay=0.01 --decay_cl=0.1 --coefficient=[0.15,0.50,0.35] --mess_dropout=0.2 --node_dropout=0.1 
```

## Suggestions for parameters
The coefficient parameter is a hyperparameter that adjusts the importance of prediction loss for each behavior.
The decay_cl is a hyperparameter the proportion of loss for self-supervised task.






