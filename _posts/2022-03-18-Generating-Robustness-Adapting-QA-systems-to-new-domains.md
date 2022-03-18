---
title: Generating Robustness - Adapting QA systems to new domains
description: A project that my team worked on for our Stanford CS224N project. We implement a variety of techniques that boost the robustness of a QA model trained with domain adversarial learning and evaluated on out-of-domain data, yielding a 16% increase in F1 score in development and 10% increase in test.
toc: true
comments: true
layout: post
categories: [projects, nlp, deep learning]
author: Nicholas
---

I would like to express my deep gratitude to my teammates (Helen Gu and Quentin Hsu) for their collaborative spirit, research ambition, and willingness to tackle open-ended problems
with tenacity. Here is our
<a href="https://drive.google.com/file/d/1-cleNk6Auyrk2rEEW7fBM30FPiLhYORX/view?usp=sharing">project report</a>
and <a href="https://drive.google.com/file/d/1qyAD_KEot7g21jRoFcN6Val0RHBb_1Z1/view?usp=sharing">poster</a>.

# Motivation
Question and Answering (QA) systems are systems that can automatically answer human questions in a natural language.
They are ubiquitous in everyday life. From the Siri voice assistant on your iPhone to intelligent chatbots, QA systems provide us with
greater convenience, allowing us to access information in an intuitive and personal manner.

The big issue is that QA systems are not robust to domain shifts, diminishing its ability to generalize to domains that it has not been trained on. Let's say that you are a government agency designing an intelligent chatbot that can answer citizens'
questions on government schemes. Citizens ask a question ("what is the age requirement for the new housing subsidy?") and the QA system extracts the answer ("21") from a corpus of information
about the policy (the "context"). The issue is that information about different policies are structured in different ways. The QA system may have been trained on context-question
pairs on housing policy and thus has a good grasp of how information on housing policy is structured, enabling it to perform efficient extractions. However, when it is faced with
context-question pairs from a new policy area (say, financial assistance schemes), it performs poorly as it has little information on the structure and characteristics of information
in that area.

In an ideal world, we would be able to plug this gap by simply getting more labelled data from the new domain and finetuning our QA model on it. However, labelled QA data
is hard to come by and extremely expensive to create (think about the number of man hours needed to create new context-question pairs). As such, we need to explore new techniques
to build robustness in QA systems, allowing them to generalize to unseen domains.

In our project, we implement a variety of techniques that boost the robustness of a QA model trained with domain adversarial learning and evaluated on out-of-domain data, yielding a **16% increase in F1 score in development** and **10% increase in test**. We find that the following innovations boost model performance: 1) finetuning the model on augmented out-of-domain data, 2) redefining domains during adversarial training to simplify the domain discriminator’s task, and 3) supplementing the training data with synthetic QA pairs generated with roundtrip consistency. We also ensemble the best-performing models on each dataset and find that ensembling yields further performance increases.

# Setup

# Baseline

