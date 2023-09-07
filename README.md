# Low-Resource Comparative Opinion Quintuple Extraction by Data Augmentation with Prompting


This repo contains two kinds of datasets, including COQE datasets and our new generated triplets dataset,  and source code of our paper.

# Datasets:
COQE datasets can refer to Liu's work (https://github.com/NUSTM/COQE)

our new generated triplets data will be made public after acceptance of the paper.

# Methods:

Different from the existing multi-stage approach, we propose an end-to-end model. In addition, we utilize the rich linguistic knowledge contained in ChatGPT to construct triples, and combine transfer learning to realize data augmentation of quintuples.

# Results:

1. The test results (F1 scores) of our approach under three kinds of matching metrics. The best performance is highlighted using bold font. 

![image-20230608171454259](https://xat20220803.oss-cn-shanghai.aliyuncs.com/Figures/202306081714496.png)

2. We further conduct a cross-domain experiment on two Chinese datasets. The results is shown as the following:

![image-20230608171514095](https://xat20220803.oss-cn-shanghai.aliyuncs.com/Figures/202306081715148.png)

3. In this paper, we are particularly curious about how ChatGPT performs on the comparative opinion  quintuple extraction task. We compare ChatGPT with our BERT-based fine-tuning method DAP. The results is shown as the following:

![image-20230608171631592](https://xat20220803.oss-cn-shanghai.aliyuncs.com/Figures/202306081716649.png)

# Code

Run different .sh files for different datasets. Each of these .sh files has two stages, the first stage is used for triple extraction and the second stage is used for prediction quintuples. The details can refer to each .sh file. 

#### Car

```
bash Car.sh
```

#### Ele

```
bash Ele.sh
```

#### Camera

```
bash Camera.sh
``` bash Camera.sh
````
