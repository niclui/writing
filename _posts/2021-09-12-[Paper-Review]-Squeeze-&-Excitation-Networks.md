---
title: Paper Review - Squeeze & Excitation Networks
description: A review of Hu et al (2017)'s seminal paper on squeeze & excitation networks (SENets). SENets allow us to model channel interdependencies, producing significant
improvements in performances at small computational cost.
toc: true
comments: true
layout: post
categories: [paper, channel attention, deep learning]
image: images/SED.png
author: Nicholas
---

_Note: This blogpost assumes that the reader has a working understanding of convolutional neural networks (CNNs).
For a good introduction to CNNs, please see this
<a href="https://towardsdatascience.com/simple-introduction-to-convolutional-neural-networks-cdf8d3077bac">article</a>._

# Introduction

## Recap of CNNs

Convolutional neural networks (CNNs) are extremely useful for wide range of visual tasks (e.g. image classification).
Here is a quick recap of how CNNs work:
- First, we pass in an image of dimensions HxWxC, where H is height, W is width, and C is the number of input channels.
For a colored image, C is usually 3 (corresponding to values for RGB). For a grayscale image, C would be 1.
- The image is then passed into a **convolutional layer** which applies a variety of filters onto the image.
These filter kernels convolve (slide) across the image, generating feature values for different parts of the image.
Each filter kernel will generate a **feature map**, which is a matrix of feature values for different parts of the image.
After applying multiple filter kernels, we will be left with **a set of feature maps, U**. U has dimensions H'XW'XC'.
The height and width of U depends on many factors (e.g. the stride of the filter kernels, or whether we have padded the image).
If we use a stride of 1 and pad the image with zeroes all around, our H' and W' will be equivalent to H and W respectively.
**C' is the number of filter kernels we applied.** It is also the **number of output channels** that we are left with.
Here is a fantastic visualisation of the process.
- Our set of feature maps, U, is then passed into a non-linear activation function (e.g. ReLU). This allows us to introduce non-linearity
into the model, allowing us to capture more complex relationships. Importantly, we also need this non-linear function to decouple
our linear layers. Intuitively, a series of linear layers is equivalent to just one linear layer. To decouple the linear layers from
one another, we insert a non-linear function in between them.
- After we have passed our feature maps, U, into the non-linear function, we pass them into a pooling layer.
The purpose of a pooling layer is to downsample the feature maps to reduce the computational load.
One example of a pooling function is max pooling, in which we convolve a NXN grid across the feature maps. We then
keep only the largest feature value inside the grid. Consequently, we are left with much smaller feature maps,
while still retaining information about the "anomalies" (e.g. parts of the image with noticeable edges).
- We go through multiple rounds of convolutional layer > non-linear function > pooling layer.
The earlier convolutional layers will capture more general patterns and relationships (e.g. edge detection).
In contrast, the later convolutional layers will capture patterns that are much more class-specific (e.g. whether the animal in the picture
has whiskers). Note that the number of output channels will grow as we go deeper into the CNN (since a feature map is passed into the next
convolutional layer and spits out multiple new feature maps).
- Finally, we pass our set of feature maps into a fully-connected layer and a final activation function to churn out a final prediction.

## Motivation for SENs 

So what exactly is sub-optimal about standard CNNs? Intuitively, there are more important feature maps and less important feature maps.
When classifying a shape, the feature map contatining information about edges may be more important than the one containing
information about background color. Thus, we will want our model to prioritize these more important feature maps, and deprioritize the less important ones.
In technical terms, we want "channel attention" - we want to give greater "attention" to the more important "channels" (aka feature maps).
Standard CNNs are unable to do this. Here is where squeeze and excitation networks come in.

# Squeeze & Excitement Networks

The seminal paper by Hu et al (2017) propose a novel neural network architecture called squeeze-and-excitement blocks.
Squeeze-and-excitement blocks allow us to model inter-dependencies between channels, and to identify which channels are more important than others.
There are three stages:

## Stage 1: Squeeze Module
We have our set of feature maps, U, which is a tensor with dimensions H'XW'XC'. We want to model the interdependencies between
the different channels/feature maps. The problem is that each channel operates within a local receptive field.
In other words, each element of a given feature map corresponds with only a specific part of the image.
This is problematic as we will want to use global spatial information when computing channel interdependencies -
if not, we will not be able to identify the interdependence between say, the value of channel 1 in the top right hand pixel, and that of channel 2 in the bottom left hand pixel.

A simple approach would be to simply use every single feature value in U. While this may improve model performance, it is extremely computationally intensive (we need to work with H'XW'XC' values and C' blows up as we go deeper into the neural network).

To mitigate this trade-off, the authors choose to generate channel-wise statistics. We can use average pooling to generate a single value for
each channel. In average pooling, we simply take the average value in a given feature map. This allows us to generate
a channel descriptor matrix of dimensions 1X1XC'. Each channel will be compressed into a single value.
We have thus squeezed our set of feature maps into a compact channel descriptor matrix that contains global spatial information.

## Stage 2: Excite Module
In this stage, we want to make use of the channel-wise information aggregated in the channel descriptor
in order to calculate the scaling weights for different channels (higher scaling weight = more important).
The authors opt for a fully connected multi-layer perceptron (MLP) bottleneck structure to map the scaling weights.
A bottleneck structure works as follows:
- We pass in our tensor of 1X1XC'.
- The input layer has C' neurons. 
- The hidden layer has C'/r neurons. Our input space is thus reduced by a factor of r. This hidden layer also introduces non-linearity with a ReLU function.
- The output layer has C' neurons. The compressed space is then expanded back to its original dimensionality.
We get back an "excited" output tensor with the same dimensions as the input tensor (1XCXC').

To maximize our cross-channel interactions, we should set r to be 1. If r > 1, we lose neurons in the hidden layer (and thus lose out on the granularity of
our cross-channel interactions). However, a smaller r also means greater computational complexity (as we have a greater number of parameters to optimize).
The default value of the reduction ratio, r, is 16.

## Stage 3: Scale Module
The excited weights tensor is then passed into a Sigmoid activation layer to scale the values to a range of 0-1.
Subsequently, by broadcasting, multiply the weights tensor with the original set of feature maps to obtain the scaled set of feature maps.
Each channel is now scaled by the weight that was learned from the MLP in stage 2.

## Where do we fit the squeeze-and-excitation block?
In standard architectures, the SE block can be inserted after the non-linearity following each convolution.

In summary, we pass in an image of HXWXC dimensions into the convolutional layer. The convolutional layer spits out
a tensor of H'XW'XC', which we pass into a non-linearity. After that, we pass the tensor into
a squeeze module which squeezes the tensor into a 1X1XC' tensor.
This 1X1XC' tensor is passed into the excite module and returns a 1X1XC' "excited" output tensor.
The excited output tensor is passed into a Sigmoid function to generate a scaled set of weights.
We then multiply these learnt weights with the set of feature maps to scale them.

# Choice of Architecture

In this section, we will evaluate the authors' architectural and hyperparameter choices.

**1. Squeeze operator:**
The authors chose to use global average pooling as the squeeze operator, instead of global max pooling.
The former takes the average value in a defined window (in our example, we take the average value across the entire feature map),
while the latter takes the maximum value in a defined window.

Max pooling allows us to preserve the most activating pixels in an image (since we preserve the highest value of the feature map).
This is especially important for things like edge detection: edges will be captured in the anomalous values of the feature map.
Max pooling allows us to preserve information about the edges of the object, while average pooling will lose out on the information.
However, it can be extremely noisy (similiar images may have very different maximum values). It also ignores the effect of neighboring pixels.

Average pooling allows us to construct a smooth average of values in a feature map. It is less noisy (similiar images won't differ by large amounts)
and takes into account neighboring pixels. The downside is that we lose information about the most activating pixels.

The authors eventually settled on average pooling as an ablation study showed that global average pooling produced a smaller error rate.

However, more recent innovations have tried to combine different methods of pooling to obtain better results.
In <a href="https://openaccess.thecvf.com/content_ECCV_2018/papers/Sanghyun_Woo_Convolutional_Block_Attention_ECCV_2018_paper.pdf">
      Convolutional Block Attention Module (CBAM)</a>, 
the input tensor is decomposed into two C'X1X1 vectors - one is generated by global average pooling, while the other is generated by global max pooling.
Global average pooling preserves aggregate spatial information, while global max pooling preserves the most activating pixels.

**2. Reduction ratio, r:**
The smaller the value of r, the more neurons we retain in the bottleneck. This allows us to capture more granular cross-channel
interactions, improving the model performance. However, it also means that we have a greater number of parameters to train, increasing computational complexity.
However, the authors find that when r decreases, performance does not increase monotonically (i.e. it may be plateauing or even decreasing at some points).
They find that r = 16 provides a good balance between accuracy and complexity.

However, note that it may not be optimal to maintain the same value of r throughout the network.
For instance, earlier convolutional layers 

**3. Excitation Operator**
An ablation study shows that Sigmoid is the best excitation operator. Other options (tanh, ReLU) significantly decreases performance.


# Overall Evaluation
