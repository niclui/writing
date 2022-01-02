---
title: Semantic segmentation of images of satellites
description: A project that my team worked on for our Stanford CS230 project. Our contributions include generating a synthetic dataset of images of unmanned spacecraft and applying state-of-the-art semantic segmentation models to obtain benchmark results.
toc: true
comments: true
layout: post
categories: [projects, semantic segmentation, deep learning]
author: Nicholas
---

I would like to express my deep gratitude to my wonderful teammates (William Armstrong and Spencer Drakontaidis) for their consistently outstanding work, strong collaborative spirit,
and willingness to tackle difficult, open-ended problems head on. In all sincerity, I could not have asked for better people to work with. Here is our
<a href="https://drive.google.com/file/d/1Dt2EqArnfQ9w1bR4G_58kiDnz9qdFdZw/view?usp=sharing">project report</a> and 
<a href="https://github.com/madwsa/mms-imageseg">github</a>.

# Motivation
The goal of semantic segmentation is to label different parts of an image with a corresponding class. We have achieved remarkable results in many domains, such
as <a href="https://paperswithcode.com/task/medical-image-segmentation">medical images</a>.

However, there has been little progress made on segmenting images of **satellites** (i.e. developing a model that can identify different satellite parts, such as
the solar panels and antenna, from a given image). This is not ideal. Developing a system that can do so is fundamental to many
important applications such as autonomous satellite rendezvous (i.e. enabling satellites to rendezvous and dock by another unmanned spacecraft with zero/little human input). 

The key obstacle is the lack of a labelled dataset of satellites. Pictures of satellites in space are hard to come by (much less labelled ones). To address this,
we generated a prototype synthetic dataset of labelled satellites and trained a variety of state-of-the-art segmentation models on it for benchmark results.

<img width="40%" alt="space" src="https://user-images.githubusercontent.com/40440105/131346647-7520b550-7ece-457d-9f23-92bb92ec5457.png">
<center><em>Soyuz-TMA satellite in space (source: Wikipedia)</em></center>

# Dataset
We use NASA's <a href="https://nasa3d.arc.nasa.gov/models">open-source 3D models of satellites</a> to produce our synthetic dataset. To provide our dataset with a variety of spacecraft configurations, we chose the Chandra X-Ray Observatory, Near Earth Asteroid Rendezvous – Shoemaker
(NEAR Shoemaker), Cluster II, and the IBEX Interstellar Boundary Explorer, as 3D models from which to generate synthetic images. We used the Blender software to process the 3D models.

## Step 1: Labelling 3D models
In consultation with an industry expert (Kevin Okseniuk, systems test engineer at Momentus), we identified eleven classes for our segmentation task.
The classes include solar panels, antennas, and thrusters. These classes were chosen because they are crucial to automous satellite rendezvous.
They include satellite parts that we should *avoid* during rendezvous (e.g. thrusters which produce liquid that may obstruct the view of the docking spacecraft)
and parts that we should *fixate on* (e.g. the bottom ring of the satellite which provides a good grip point for the docking spacecraft).

Using Blender, we then labelled each part of the satellite with a unique color.

<img width="60%" alt="space" src="https://user-images.githubusercontent.com/40440105/147863638-d51398ec-c73b-464a-ba4d-d4750f7372a4.png">
<center><em>Unlabelled vs Labelled satellite image</em></center>

## Step 2: Artistic modifications
We compose a series of artistic modifications to make the 3D models look more realistic. For instance, we simulated the lighting conditions of
<a href="https://en.wikipedia.org/wiki/Low_Earth_orbit">low earth orbit</a> by illuminating the model with two light sources: one light source at infinity simulating the intensity,
color, and parallel light rays of the sun, and one planar light source to simulate earthshine, i.e. the sunlight reflected by the surface of the earth.

## Step 3: Generating synthetic dataset
We wrote a Python script to move the camera in Blender in a spherical pattern around the
3D model to one of 5000 positions. For each position,
three rendered images were generated with the same
aspect, but with different ranges. This gave us 15,000 images for each 3D model and a total of 60,000 images.

This process was repeated for both the unlabelled and labelled 3D models, giving us 60,000 base image and ground truth pairs.

## Step 4: Ground truth representation
Originally, for a given synthetic image, each pixel has three values for its R/G/B colors.
To make these images more understandable for our model, we used Python's Pillow library
to map each combination of RGB values to the corresponding class label (ranging from 0 to 10).
Subsequently, each pixel of a synthetic image contains only one value (from 0 to 10) which corresponds to its respective class.

# Model Training
## Architectures
After preparing our synthetic dataset, we proceed with training 3 state-of-the-art deep learning segmentation models using Python's FastAI and SemTorch libraries.
In each case, a backbone pre-trained on ImageNet was incorporated to leverage transfer learning in extracting features from the input image. 

### I. U-Net
<a href="https://medium.com/@keremturgutlu/semantic-segmentation-u-net-part-1-d8d6f6005066">U-Net</a> is an encoder-decoder network which aims to provide precise localization even when using a
smaller dataset than is typically used for image segmentation tasks. 

### II. HRNet
<a href="https://towardsdatascience.com/hrnet-explained-human-pose-estimation-sematic-segmentation-and-object-detection-63f1ce79ef82">HRNet</a> (High-Resolution Net) is a CNN developed specifically to retain and use high-resolution inputs
throughout the network, resulting in better performance for segmentation tasks.
HRNet aims to provide high spatial precision, which is desirable in this task due to the variety of classes and class imbalance.

### III. DeepLab
<a href="https://towardsdatascience.com/the-evolution-of-deeplab-for-semantic-segmentation-95082b025571">DeepLab</a> is a CNN developed and open-sourced by Google that relies heavily on
<a href="https://towardsdatascience.com/review-deeplabv3-atrous-convolution-semantic-segmentation-6d818bfd1d74">Atrous Convolution</a>
to perform image segmentation tasks.
More specifically, we used the latest iteration of the DeepLab model at time of writing, DeepLabv3+, as implemented by FastAI.

## Loss Functions
We also experimented with a variety of loss functions to mitigate the class imbalance in the dataset (for instance, the background/non-essential satellite parts
takes up ~94% of all pixels).

### I. Categorical Cross-Entropy Loss
For each pixel, this function computes the log loss
summed over all possible classes.

$Loss_i = - \sum_{classes} y \log(\hat{y})$

This scoring is computed over all pixels and the average taken. However, this loss function is susceptible to
class imbalance. For unbalanced data, training might be dominated by the most prevalent class.

### II. Dice Score
For a given pixel, we compute the F1 score (also known as the Dice Coefficient) for all 11 classes.
The Dice Score is given by 1 minus the arithmetic mean across all 11 classes.
We are able to mitigate class imbalance as the F1 score balances between precision and recall.

### III. Dice Score + Focal Loss
Focal loss modifies the pixel-wise cross-entropy loss by down-weighting the loss of easy-to-classify pixels based on a hyperparamter $\gamma$, focusing training on more difficult examples. The focal loss is given by:

$FocalLoss_i = - \sum_{classes} (1 - \hat{y})^{\gamma} y \log(\hat{y})$

Dice + focal loss blends Dice and focal loss with a mixing parameter α applied to the focal loss, balancing
global (Dice) and local (focal) features of the target mask. We used the default values of $\gamma$ = 2 and $\alpha$ = 1 during training

# Results

# Conclusion


