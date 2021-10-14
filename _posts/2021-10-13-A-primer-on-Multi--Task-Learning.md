---
title: A primer on multi-task learning
description: In multi-task learning, we train a generalist model that is able to perform several different tasks. By exploiting similiarities between the different tasks, multi-task learning is able to achieve better performance than individual models for each task.
toc: true
comments: true
layout: post
categories: [multi-task learning, deep learning]
image: images/HARD.png
author: Nicholas
---
Hi everyone! Today I would like to share about a new topic in ML that I have been exploring: multi-task learning (MTL). In MTL, we train a “generalist” model that is able to perform a variety of tasks. By exploiting similarities between the different tasks, we are able to achieve **better performance more quickly** for each individual task. Curious to see how this works? Let’s dive in!

# Overview
Let’s say that we need to perform 3 different tasks - detecting a dog, detecting a cat, and detecting a giraffe. The standard approach would be to train individual CNNs for each task. However, we may run into data limitations. Neural networks perform very well on large and diverse datasets. However, our dataset for each task may be small, leading to poor performance for each individual task. Intuitively, if our model has only a few dog pictures to learn from, it may not be able to differentiate between important dog-specific features and irrelevant features. This may lead to us over-fitting on irrelevant features, hurting the model’s generalizability.

Multi-task learning is able to circumvent these limitations. Instead of having one model for each task, we build a “generalist” model that performs all three tasks simultaneously. This is because the different tasks share similarities. For instance, both dog and cat detection will require the model to identify the presence of fur. By exploiting similarities between the different tasks, we are able to achieve good performance even with limited datasets for each task. Let us first go through the technical details of how MTL exploits task similarities through parameter sharing. Afterwards, we will give an intuitive illustration of how this improves performance.

# Parameter Sharing
There are two types of parameter sharing - hard parameter sharing (where every task shares a large number of common parameters) and soft parameter sharing (where we constrain the parameters of different tasks, ensuring that they are similar to each other). This encourages the model to learn a general representation across the different tasks, reducing the risk of us overfitting on any one task.

## Hard Parameter Sharing
We will explore the most basic ways to introduce hard parameter sharing, which is the most common approach in MTL.

### Method 1: Multi-head architecture
One simple way is to design an architecture where the hidden layers are shared among all the tasks. Only the output layer is task-specific and we have multiple heads corresponding to each task.

<img width="40%" alt="hard" src="https://user-images.githubusercontent.com/40440105/137263811-f9c8a679-4d2b-47d4-869f-aef1c156bc8b.png">
<center><em>Source: Ruder (2017)</em></center>

### Method 2: Conditioning on task descriptor vector
We can create a task descriptor vector, which contains information about the different tasks. For instance, we can create a one-hot vector of the task index. Task 1 will correspond to [1,0,0,...], task 2 will correspond to [0,1,0,...] and so on. By conditioning on this vector, we are able to control where and how we split the set of parameters into common parameters and task-specific ones.

The simplest form of conditioning is concatenation-based. Consider the following CNN. In this set-up, the convolutional layers are shared among all the tasks.
However, at the first fully connected layer, we can concatenate the input vector with the task index vector (denoted as z).

<img width="80%" alt="cnn" src="https://user-images.githubusercontent.com/40440105/137263748-066e1b44-7ec9-4b95-bcbf-baa1ac44002e.png">
<center><em>Source: Stanford CS330 Course</em></center>

In other words, we stack x on top of z. Correspondingly, we expand the weights matrix of the FC layer.

<img width="40%" alt="concat" src="https://user-images.githubusercontent.com/40440105/137263678-34eda649-50ee-4017-b471-afb28c9c7e5b.png">
<center><em>Source: Stanford CS330 Course</em></center>

This weights matrix will now contain both common parameters and task-specific ones. Parameters in columns 1 to D are shared across all tasks. In contrast, parameters in columns D+1 to T are task-specific. To see this, consider parameters in column D+1. These parameters will be multiplied with z1 in matrix multiplication. If z1 is one (i.e. task 1), these parameters will be “activated”. If z1 is zero (i.e. any task other than task 1), these parameters will be multiplied with a zero value. As such, parameters in column D+1 are specific to task 1. Column D+2 is specific to task 2, and so on and so forth.

<img width="80%" alt="concat" src="https://user-images.githubusercontent.com/40440105/137265201-eab44868-2235-4dca-83ad-de245c4e0e80.jpg">
<center><em>Source: Personal drawing</em></center>


Overall, we have a model that shares the hidden layers across the tasks. In the fully-connected layers, there is a mix of shared parameters and task-specific ones. By varying the location where we introduce the task index vector, we are able to control where the split of parameters happen.

There are other options such as multiplicative-based conditioning. Instead of stacking x on top of z, we can use the product of x and z. This allows us to capture interaction effects - the weight applied to x differs based on the value of z. The choice of z is another important consideration. Instead of using a one-hot vector of the task index, we can consider choosing a descriptor vector that contains metadata of the different tasks. This will allow us to better exploit similarities between the different task structures.  

## Soft Parameter Sharing
In soft parameter sharing, each task has its own model with its own parameters. However, to encourage cross-learning between the different tasks, we impose regularization. The regularization penalty (e.g. L2 norm between the parameter vectors of different models) forces the parameters of the different tasks to be close to each other, ensuring that the model learns a more general representation across the different tasks.

<img width="80%" alt="cnn" src="https://user-images.githubusercontent.com/40440105/137263845-733fe448-f3fc-4c80-86e5-3b2b62e1c720.png">
<center><em>Source: Ruder (2017)</em></center>

## Hard vs Soft Parameter Sharing
So which type of parameter sharing is better? It depends. Hard parameter sharing greatly reduces the risk of overfitting on a specific task. Intuitively, given that the different tasks share a significant % of the set of parameters, our model is more likely to find a representation that generalizes across all the tasks.

The downside of hard parameter sharing is its lack of flexibility. We impose a large number of common parameters for the different tasks. This is potentially problematic - let’s say that we have a feature (e.g. presence of a snout) that is more useful for dog detection than giraffe detection. For the best results, we want to attach a heavier weight to this feature for dog detection (and a smaller weight for giraffe detection). However, in hard parameter sharing, we impose the same weight for all classes, affecting the model’s performance.

Soft parameter sharing provides a compromise between this trade-off. It is slightly more prone to overfitting, but allows for greater flexibility in parameter choice.
# Why MTL works
MTL tries to exploit the similarities between different tasks. There are a few mechanisms through which this exploitation can improve model performance and reduce training time:

**(I) Implicit data augmentation:** MTL effectively allows us to expand our dataset for any given task. All tasks are somewhat noisy. For instance, the dataset for any given task is never perfect - it may contain a few wrong labels or low quality images. In MTL, we train on multiple tasks. Assuming that the noise patterns of different tasks are unrelated, we are able to average the noise patterns across a variety of tasks.

**(II) Attention focusing:** If we have an extremely noisy task (e.g. a large percentage of pictures are blurry), there is a high risk of overfitting on less important features. MTL can help the model focus its attention on more important features as other tasks will provide additional evidence for the relevance or irrelevance of those features.

**(III) Eavesdropping:** Some features G may be easy to learn for task B, while being difficult to learn for task A. This may be because A interacts with the features in a more complex or less clear way. Through MTL, we can allow the model to “eavesdrop” (i.e. learn G through task B).

Consider the following two tasks: dog breed detection and giraffe breed detection. One feature that is useful for both tasks is the neck length (the neck length of dogs and giraffes vary across breeds). We will certainly learn this feature in the giraffe breed detection task given how prominent their necks are and the large variation across giraffe breeds. However, we are less likely to do so in the dog breed detection task as differences in neck length are less obvious. To circumvent this, we train the model on both tasks, increasing the probability that we capture this feature for dog breed detection too.

There is a subtle difference between (II) and (III). We can think of (III) as being a more extreme version of (II). In (II), the single-task model can pick up important features but is not sure how important they are. In (III), the single-task model may miss out on these features entirely.

**(IV) Representation bias:** MTL prefers a generalized representation that extends across multiple tasks. This will allow us to learn new tasks more easily in the future (as long as those new tasks share commonalities with our existing set of tasks).


# Negative Transfer
Negative transfer occurs when training individual models on multiple tasks produces better performance than a single MTL model. There are two general cases when this arises:

Firstly, if the tasks are unrelated to each other (e.g. dog detection vs text sentiment analysis). In which case, there are no similarities between the different tasks for MTL to exploit. There may even be a decrease in performance since MTL imposes shared parameters across the different tasks. It would be better to give these distinct tasks full control over their own set of parameters.

Secondly, if the information learnt in one task contradicts that of another task. Consider a dog detection and plant detection task. For the dog detection task, the presence of fur is extremely important. In contrast, for the plant detection task, that feature is irrelevant. If we train the model on both tasks, we can expect to attach a moderate weight to that feature. This weight will be less than the weight that we would learn from training our model on just dog detection. And more than the weight we would learn from training our model on just plant detection. Consequently, the MTL weight is suboptimal for both dog and plant detection.

# Conclusion
That’s the end of our primer on multi-task learning! MTL is a powerful and promising technique that allows us to exploit similarities between different related tasks. It does so through parameter sharing. It is important to be discerning about the tasks we include - if not, we may run into the problem of negative transfer (where it is better to train individual models).

# Useful Resources
- <a href = "https://cs330.stanford.edu/">Stanford CS330 lectures</a>
- <a href = "https://arxiv.org/abs/1706.05098">Review paper by Ruder (2017)</a>
- <a href = "https://ieeexplore.ieee.org/abstract/document/9392366">Review paper by Zhang and Yang (2021)</a> (note: more technical than Ruder (2017))
