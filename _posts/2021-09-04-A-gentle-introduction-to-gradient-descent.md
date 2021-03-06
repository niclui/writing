---
title: A gentle introduction to gradient descent
description: An intuitive overview of key concepts in gradient descent.
toc: true
comments: true
layout: post
categories: [gradient descent, deep learning]
image: images/skye.png
author: Nicholas
---

Gradient descent is one of the key foundations of machine learning. Today, I hope to give a high-level and intuitive overview of this concept.

# Gradient descent
## Background
Recall the process of training a machine learning model. The steps broadly involve:
- Initializing a bunch of parameters
- Using those parameters to make predictions from your input
- Based on the quality of your predictions, tweaking your parameters to make the model better
- Repeating steps 2 and 3 until you are satisfied with your model (e.g. when the overall prediction error is below a certain threshold).

Gradient descent (GD) is an algorithm that helps us accomplish these steps.
To illustrate this concept, let us imagine that we are building a model which can help us distinguish between two handwritten digits - say, the numbers 3 and 7.
We are training our model on a large dataset of handwritten 3s and 7s - for simplicity, the pictures are all grayscale and have the same dimensions.

<img width="40%" alt="handwritten37" src="https://user-images.githubusercontent.com/40440105/131346647-7520b550-7ece-457d-9f23-92bb92ec5457.png">
<center><em>Source: 1001 Free Downloads</em></center>

## How does gradient descent work?
**Step 1: Initializing a bunch of parameters**

Our parameters can be the weights attached to every feature of the picture (lets say that we have 100 features). We start off by assigning a random weight to each feature.

**Step 2: Using those parameters to make predictions from your input**

We can multiply the value of every feature with its weight. After that, get the average weighted value across all features.
Following which, pass that value into a special function (e.g. Sigmoid function) to generate a probability between 0 and 1.
If that probability is >0.5, predict a 3. If not, predict a 7.

**Step 3: Tweaking your parameters**

The magic of gradient descent happens here. Intuitively, we want to update our parameters in a way that will make our predictions better.

Let's imagine that we are updating the weight of feature m. We plot this weight on the x-axis.
On the y-axis, we will plot the **loss function** with respect to that weight. What is the loss function?
Essentially, it is a function that will return a small value when your model is good, and a large value when your model is bad.
An example of a loss function is <a href="https://www.analyticsvidhya.com/blog/2021/03/binary-cross-entropy-log-loss-for-binary-classification/">binary cross entropy</a>. This loss function will give us a small value when we are making a correct prediction with high confidence, a medium value when we are making a correct prediction with poor confidence, and a large value when we are making a wrong prediction.

Gradient descent involves us taking iterative steps to find the local minima of the loss function (with respect to the parameter we are adjusting).
Consider the case of a convex loss function (see picture below). Let's say that we initialize our weight at a value of 1. The GD algorithm will calculate the gradient at that point.
To calculate the gradient, we can adjust the weight by a tiny margin (holding other weights constant) and see the impact that it has on the loss function.
By dividing the change in loss by the change in weight, we get the gradient.

The algorithm will then increase/decrease the value of x by the value of the gradient multiplied by a specified learning rate. Since the gradient is negative (i.e. we will
reduce our loss by increasing our weight), the algorithm will increase the weight by the value of gradient * the learning rate.

<img width="40%" alt="gd1" src="https://user-images.githubusercontent.com/40440105/131346851-985a63e7-0013-4f05-9ea5-a89d31d74c2d.png">
<center><em>Source: fast.ai</em></center>

> Tip: The learning rate controls the rate at which the model adjusts its parameters. Selecting the optimal learning rate is tricky. If we select an overly large rate, the model will adjust the parameters by huge amounts, potentially resulting in us bypassing the local minima. In contrast, if the learning rate is too small, it will take a long time to reach the local minima.
> There are several ways to tune this important hyperparameter.
> For instance, you could do learning rate annealing. Start off with a high learning rate so that you can quickly descend to an acceptable set of parameter values.
> After that, decrease your learning rate so that you can precisely locate the optimal value within the acceptable range.
> Here's a [good article](https://automaticaddison.com/how-to-choose-an-optimal-learning-rate-for-gradient-descent/) which explains more.

**Step 4: Iterate until you are satisfied**

Eventually, after multiple rounds of iteration, we will reach the local minima of the loss function. At this point, our gradient is zero and any further adjustments to the weight will increase loss.

<img width="40%" alt="gd2" src="https://user-images.githubusercontent.com/40440105/131346913-c43a5d47-42c7-4519-9895-882e436ff595.png">
<center><em>Source: fast.ai</em></center>

We repeat this process for every weight (all 100 of them!).

A common analogy for gradient descent is that of a blindfolded hiker who is stuck on the side of a hill. He wants to get to as low a point as possible.
Thus, he feels the ground around him and takes a small step in the steepest downward direction. This is one iteration. By taking many of these small steps, he will
eventually reach the bottom of a valley (local minima).

<img width="40%" alt="skye" src="https://user-images.githubusercontent.com/40440105/131346963-4006c387-4d78-4d51-ba93-08d21c03615c.png">
<center><em>Source: inspiredbymaps</em></center>

> Tip: Note that gradient descent requires our loss function to be continuous and smooth. 
> What if our loss function is discontinuous - say, a step function?
> To illustrate this, initialize our weight at 1. The gradient at that point is completely flat. Thus, the GD algorithm will terminate immediately.
> However, if we increased our weight by a larger amount, we could have moved to a lower step of the loss function. In this instance, the GD algorithm will not give us good results and the model will not learn well.
> This is why we cannot use accuracy (the % of correct classifications) as our loss function. We can imagine accuracy to be represented by a step function - if we change
> a weight by a tiny amount, we do not expect any prediction to change from a 3 to 7 (or vice versa). As such, accuracy remains unchanged.
> A much larger adjustment in a weight is needed to induce a change in our predictions.
> Consequently, we use a continuous loss function which improves when we make correct predictions with slightly more confidence, or make wrong predictions with slightly less confidence.
> 
> <img width="362" alt="desmos_step" src="https://user-images.githubusercontent.com/40440105/131347027-3e85cdf4-4d61-440f-a312-dd44d18e1051.png">
> 
> <center><em>Source: Desmos</em></center>


## What are the limitations of gradient descent?

There are two key limitations behind standard gradient descent (or batch gradient descent):
- Firstly, it is possible that we can get stuck in a local minima of the loss function, preventing us from accessing the better global minima. Consider the illustration below. We start at point U and adjust iteratively until we hit the local minima and the GD algorithm terminates.

<img width="360" alt="minima2" src="https://user-images.githubusercontent.com/40440105/131347079-b0cb5ead-d0f6-4d37-822c-bdc5a7564d14.png">
<center><em>Source: Analytics Vidhya</em></center>

- Secondly, in standard gradient descent, we use every single datapoint in our training set to compute gradients. Let's say we have 5,000 training images. For a given parameter, we use all 5,000 images to calculate individual losses and take the mean. After that, we adjust the parameter value slightly and re-calculate the loss for all 5,000 images (taking the mean again). The difference in means divided by the difference in weight is the gradient. When the training set is large, this process becomes computationally intensive.

# Other flavours of gradient descent
## Stochastic gradient descent

To remedy those limitations, stochastic gradient descent is a popular alternative.
In stochastic gradient descent, we do not use the entire training dataset in our computation of gradients. Instead, in each iteration, we randomly select a training datapoint to do so.
This gives us a stochastic (i.e. random) approximation to the gradient calculated with the entire training dataset. By constantly iterating, the parameter value should
move in the same general direction (as standard gradient descent), while being computationally more efficient.

The randomness can also allow us to escape local minima. Every training datapoint has its own unique loss function. In standard gradient descent, we average all these
loss functions into one aggregate loss function. In unfortunate cases, we will
move down that average loss function into a sub-optimal local minima where we are trapped forever.

However, in stochastic gradient descent, we can potentially avoid this negative outcome. In each iteration, we hop from the loss function of one training datapoint
to another. Even if we get stuck in the local minima of a loss function, we can move to a new loss function in the next iteration (where our current parameter value is not
in a local minima anymore). This allows us to keep moving and iterating.

Stochastic gradient descent faces two key weaknesses:
- The stochastic steps it take can be very noisy. This may result in us taking more time to converge 
to the local minima of the loss function (ignoring the case where the local minima may be sub-optimal). Stochastic gradient descent is computationally more efficient, but may end up taking more time.
- Computing infrastructure in ML (e.g. GPUs) are optimised for vectorized operations (vector addition, multiplication). Given that we use only one training datapoint at a time, we give up this powerful capability.


## Mini-batch gradient descent

Mini-batch gradient descent is a compromise between standard gradient descent and stochastic gradient descent. We do not use the entire training dataset or a single training datapoint.
Instead, we use small batches (typically ~30-500) of training datapoints to compute our gradient. Given that we use small batches of datapoints at a time, we are also able to harness the performance of GPUs in vectorized operations.

# Conclusion

I hope this article has given you a good overview of the key concepts in gradient descent! Here's a quick summary:

- Gradient descent is an algorithm that allows us to iteratively adjust our model's parameters for better performance. We adjust our parameters in a direction that brings us down the loss function.
- The two biggest limitations of standard gradient descent are 1) the possibility that we may be trapped in the local minima of our loss function, and 2) the computational cost.
- Stochastic and mini-batch gradient descent are popular alternatives that mitigate some of these issues.

Hope you enjoyed reading!

(Cover picture credit: inspiredbymaps)
