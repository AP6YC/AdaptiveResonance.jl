# Background

This page provides a theoretical overview of Adaptive Resonance Theory and what this project aims to accomplish.

## What is Adaptive Resonance Theory?

Adaptive Resonance Theory (commonly abbreviated to ART) is both a **neurological theory** and a **family of neurogenitive neural network models for machine learning**.

ART began as a neurocognitive theory of how fields of cells can continuously learn stable representations, and it evolved into the basis for a myriad of practical machine learning algorithms.
Pioneered by Stephen Grossberg and Gail Carpenter, the field has had contributions across many years and from many disciplines, resulting in a plethora of engineering applications and theoretical advancements that have enabled ART-based algorithms to compete with many other modern learning and clustering algorithms.

Because of the high degree of interplay between the neurocognitive theory and the engineering models born of it, the term ART is frequently used to refer to both in the modern day (for better or for worse).

Stephen Grossberg's has recently released a book summarizing the work of him, his wife and colleague Gail Carpenter, and his other colleagues on Adaptive Resonance Theory in his book [Conscious Brain, Resonant Mind](https://www.amazon.com/Conscious-Mind-Resonant-Brain-Makes/dp/0190070552).

## ART Basics

![art](../assets/figures/art.png)

### ART Dynamics

Nearly every ART model shares a basic set of dynamics:

1. ART models typically have two layers/fields denoted F1 and F2.
2. The F1 field is the feature representation field.
    Most often, it is simply the input feature sample itself (after some necessary preprocessing).
3. The F2 field is the category representation field.
    With some exceptions, each node in the F2 field generally represents its own category.
    This is most easily understood as a weight vector representing a prototype for a class or centroid of a cluster.
4. An activation function is used to find the order of categories "most activated" for a given sample in F1.
5. In order of highest activation, a match function is used to compute the agreement between the sample and the categories.
6. If the match function for a category evaluates to a value above a threshold known as the vigilance parameter ($$\rho$$), the weights of that category may be updated according to a learning rule.
7. If there is complete mismatch across all categories, then a new categories is created according to some instantiation rule.

### ART Considerations

In addition to the dynamics typical of an ART model, you must know:

1. ART models are inherently designed for unsupervised learning (i.e., learning in the absense of supervisory labels for samples).
    This is also known as clustering.
2. ART models are capable of supervised learning and reinforcement learning through some redesign and/or combination of ART models.
    For example, ARTMAP models are combinations of two ART models in a special way, one learning feature-to-category mappings and another learning category-to-label mappingss.
    ART modules are used for reinforcement learning by representing the mappings between state, value, and action spaces with ART dynamics.
3. Almost all ART models face the problem of the appropriate selection of the vigilance parameter, which may depend in its optimality according to the problem.
4. Being a class of neurogenitive neural network models, ART models gain the ability for theoretically infinite capacity along with the problem of "category proliferation," which is the undesirable increase in the number of categories as the model continues to learn, leading to increasing computational time.
    In contrast, while the evaluation time of a fixed architecture deep neural network is always *exactly the same*, there exist upper bounds in their representational capacity.
5. Nearly every ART model requires feature normalization (i.e., feature elements lying within $$[0,1]$$) and a process known as complement coding where the feature vector is appended to its vector complement $$[1-\bar{x}]$$.
   This is because real-numbered vectors can be arbitrarily close to one another, hindering learning performance, which requires a degree of contrast enhancement between samples to ensure their separation.

To learn about their implementations, nearly every practical ART model is listed in a recent [ART survey paper by Leonardo Enzo Brito da Silva](https://arxiv.org/abs/1905.11437).

## History and Development

At a high level, ART began with a neural network model known as the Grossberg Network named after Stephen Grossberg.
This network treats the firing of neurons in frequency domain as basic shunting models, which are recurrently connected to increase their own activity while suppressing the activities of others nearby (i.e., on-center, off-surround).
Using this shunting model, Grossberg shows that autonomous, associative learning can occur with what are known as instar networks.

By representing categories as a field of instar networks, new categories could be optimally learned by the instantiation of new neurons.
However, it was shown that the learning stability of Grossberg Networks degrades as the number of represented categories increases.
Discoveries in the neurocognitive theory and breakthroughs in their implementation led to the introduction of a recurrent connections between the two fields of the network to stabilize the learning.
These breakthroughs were based upon the discovery that autonomous learning depends on the interplay and agreement between *perception* and *expectation*, frequently referred to as bottom-up and top-down processes.
Furthermore, it is *resonance* between these states in the frequency domain that gives rise to conscious experiences and that permit adaptive weights to change, leading to the phenomea of attention and learning.
The theory has many explanatory consequences in psychology, such as why attention is required for learning, but its consequences in the engineering models are that it stabilizes learning in cooperative-competitive dynamics, such as interconnected fields of neurons, which are most often chaotic.

Chapters 18 and 19 of the book by [Neural Network Design by Hagan, Demuth, Beale, and De Jesus](https://hagan.okstate.edu/NNDesign.pdf) provide a good theoretical basis for learning how these network models were eventually implemented into the first binary-vector implementation of ART1.
