
# 1.1.0

* Requirements now points to 1.5.0 TensorFlow

# 1.0.9

* Residual block in CycleGAN was not using first convolutional layer

# 1.0.8

* Batch loading support from magenta repo for FastGen config

# 1.0.7

* NSynth batch processing code from magenta repo
* `get_model` function in `nsynth` module now attempts to download and untar the model from the magenta website.
* `utils.download` functions default to local dir
* Separate encode functionality in nsynth module.

# 1.0.6

* MDN activation fn

# 1.0.5

* Fix gauss pdf

# 1.0.4

* Allow for batch=1 in DRAW code

# 1.0.3

* Add mdn to init

# 1.0.2

* Remove tanh activation from variational layer
* Add librispeech train code to fastwavenet module
* Add Mixture Density Network code from course in mdn module

# 1.0.1

Fixed model loading during charrnn infer method.  No longer checks for ckpt name and will attempt to load regardless.

# 1.0.0

Initial release
