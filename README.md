# OVAE

# Continual Variational Autoencoder Learning via Online Cooperative Memorization

>📋 This is the implementation of Continual Variational Autoencoder Learning via Online Cooperative Memorization

>📋 Accepted by ECCV 2022

# Title : Continual Variational Autoencoder Learning via Online Cooperative Memorization

# Paper link  



# Abstract

Due to their inference, data representation and reconstruction properties, Variational Autoencoders (VAE) have been successfully used in continual learning classification tasks. However, their ability to generate images with specifications corresponding to the classes and databases learned during Continual Learning (CL) is not well understood and catastrophic forgetting remains a significant challenge. In this paper, we firstly analyze the forgetting behaviour of VAEs by developing a new theoretical framework that formulates CL as a dynamic optimal transport problem. This framework proves approximate bounds to the data likelihood without requiring the task information and explains how the prior knowledge is lost during the training process. We then propose a novel memory buffering approach, namely the Online Cooperative Memorization (OCM) framework, which consists of a Short-Term Memory (STM) that continually stores recent samples to provide future information for the model, and a Long-Term Memory (LTM) aiming to preserve a wide diversity of samples. The proposed OCM transfers certain samples from STM to LTM according to the information diversity selection criterion without requiring any supervised signals. The OCM framework is then combined with a dynamic VAE expansion mixture network for further enhancing its performance.

# Environment

1. Tensorflow 2.1
2. Python 3.6

# Training and evaluation

>📋 Python xxx.py, the model will be automatically trained and then report the results after the training.

>📋 Different parameter settings of LMix would lead different results and we also provide different settings used in our experiments.

# BibTex
>📋 If you use our code, please cite our paper as:


