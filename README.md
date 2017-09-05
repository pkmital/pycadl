# Introduction
This package is part of the Kadenze Academy program [Creative Applications of Deep Learning w/ TensorFlow](https://www.kadenze.com/programs/creative-applications-of-deep-learning-with-tensorflow).

[COURSE 1: Creative Applications of Deep Learning with TensorFlow I](https://www.kadenze.com/courses/creative-applications-of-deep-learning-with-tensorflow-iv/info) (Free to Audit)  
Session 1: Introduction to TensorFlow  
Session 2: Training A Network W/ TensorFlow  
Session 3: Unsupervised And Supervised Learning  
Session 4: Visualizing And Hallucinating Representations  
Session 5: Generative Models  

[COURSE 2: Creative Applications of Deep Learning with TensorFlow II](https://www.kadenze.com/courses/creative-applications-of-deep-learning-with-tensorflow-ii/info) (Program exclusive)  
Session 1: Cloud Computing, GPUs, Deploying  
Session 2: Mixture Density Networks  
Session 3: Modeling Attention with RNNs, DRAW  
Session 4: Image-to-Image Translation with GANs  

[COURSE 3: Creative Applications of Deep Learning with TensorFlow III](https://www.kadenze.com/courses/creative-applications-of-deep-learning-with-tensorflow-iii-iii/info) (Program exclusive)  
Session 1: Modeling Music and Art: Google Brain’s Magenta Lab  
Session 2: Modeling Language: Natural Language Processing  
Session 3: Autoregressive Image Modeling w/ PixelCNN  
Session 4: Modeling Audio w/ Wavenet and NSynth  

# Requirements

Python 3.5+

# Installation

`pip install cadl`

Then in ptyhon, you can import any module like so:

`from cadl import vaegan`

Or see a list of possible modules in an interactive console by typing:

`from cadl import ` and then pressing tab to see the list of available modules.

# Documentation

[cadl.readthedocs.io](http://cadl.readthedocs.io)

# Contents 

This package contains various models, architectures, and building blocks covered in the Kadenze Academy program including:

* Autoencoders  
* Character Level Recurrent Neural Network (CharRNN)  
* Conditional Pixel CNN  
* CycleGAN  
* Deep Convolutional Generative Adversarial Networks (DCGAN)  
* Deep Dream  
* Deep Recurrent Attentive Writer (DRAW)  
* Gated Convolution  
* Generative Adversarial Networks (GAN)  
* Global Vector Embeddings (GloVe)  
* Illustration2Vec  
* Inception  
* Mixture Density Networks (MDN)  
* PixelCNN  
* NSynth  
* Residual Networks 
* Sequence2Seqeuence (Seq2Seq) w/ Attention (both bucketed and dynamic rnn variants available)  
* Style Net  
* Variational Autoencoders (VAE)  
* Variational Autoencoding Generative Adversarial Networks (VAEGAN)  
* Video Style Net  
* VGG16  
* WaveNet / Fast WaveNet Generation w/ Queues / WaveNet Autoencoder (NSynth)  
* Word2Vec  

and more.  It also includes various datasets, preprocessing, batch generators, input pipelines, and plenty more for datasets such as:

* CELEB  
* CIFAR  
* Cornell  
* MNIST  
* TedLium  
* LibriSpeech  
* VCTK  

and plenty of utilities for working with images, GIFs, sound (wave) files, MIDI, video, text, TensorFlow, TensorBoard, and their graphs.

Examples of each module's use can be found in the tests folder.

# Contributing

Contributions, such as other model architectures, bug fixes, dataset handling, etc... are welcome and should be filed on the GitHub.
