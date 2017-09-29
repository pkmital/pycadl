"""Mixture Density Network.
"""
"""
Copyright 2017 Parag K. Mital.  See also NOTICE.md.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.contrib.layers as tfl
from skimage.data import astronaut, coffee
from scipy.misc import imresize


def gausspdf_np(x, mean, sigma):
    return np.exp(-(x - mean)**2 /
                  (2 * sigma**2)) / (np.sqrt(2.0 * np.pi) * sigma)


def gausspdf(x, mean, sigma):
    return tf.exp(-(x - mean)**2 /
                  (2 * sigma**2)) / (tf.sqrt(2.0 * np.pi) * sigma)


def build_single_gaussian_model(n_input_features=2,
                                n_output_features=3,
                                n_neurons=[128, 128]):
    X = tf.placeholder(tf.float32, shape=[None, n_input_features], name='X')
    Y = tf.placeholder(tf.float32, shape=[None, n_output_features], name='Y')

    current_input = X
    for layer_i in range(len(n_neurons)):
        current_input = tfl.linear(
            inputs=current_input,
            num_outputs=n_neurons[layer_i],
            activation_fn=tf.nn.tanh,
            scope='layer/' + str(layer_i))

    means = tfl.linear(
        inputs=current_input,
        num_outputs=n_output_features,
        activation_fn=None,
        scope='means')
    sigmas = tf.maximum(
        tfl.linear(
            inputs=current_input,
            num_outputs=n_output_features,
            activation_fn=tf.nn.relu,
            scope='sigmas'), 1e-10)

    p = gausspdf(Y, means, sigmas)
    negloglike = -tf.log(tf.maximum(p, 1e-10))
    cost = tf.reduce_mean(tf.reduce_mean(negloglike, 1))
    return X, Y, cost, means


def get_data(img):
    xs = []
    ys = []
    for row_i in range(img.shape[0]):
        for col_i in range(img.shape[1]):
            xs.append([row_i, col_i])
            ys.append(img[row_i, col_i])

    xs = np.array(xs)
    ys = np.array(ys)
    xs = (xs - np.mean(xs)) / np.std(xs)
    ys = (ys / 255.0)
    return xs, ys


def train_single_gaussian_model():
    img = imresize(astronaut(), (64, 64))
    xs, ys = get_data(img)
    n_iterations = 500
    batch_size = 50
    fig, ax = plt.subplots(1, 1)
    with tf.Graph().as_default() as g, tf.Session(graph=g) as sess:
        X, Y, cost, means = build_single_gaussian_model()
        optimizer = tf.train.AdamOptimizer(0.001).minimize(cost)
        init_op = tf.global_variables_initializer()
        # Here we tell tensorflow that we want to initialize all
        # the variables in the graph so we can use them
        # This will set W and b to their initial random normal value.
        sess.run(init_op)
        # We now run a loop over epochs
        for it_i in range(n_iterations):
            idxs = np.random.permutation(range(len(xs)))
            n_batches = len(idxs) // batch_size
            for batch_i in range(n_batches):
                idxs_i = idxs[batch_i * batch_size:(batch_i + 1) * batch_size]
                sess.run(optimizer, feed_dict={X: xs[idxs_i], Y: ys[idxs_i]})
            this_cost = sess.run([cost], feed_dict={X: xs, Y: ys})
            print('cost:', this_cost)
            if (it_i + 1) % 20 == 0:
                ys_pred = means.eval(feed_dict={X: xs}, session=sess)
                img = np.clip(ys_pred.reshape(img.shape), 0, 1)
                plt.imshow(img)
                plt.show()
                fig.canvas.show()


def build_multiple_gaussians_model(n_input_features=2,
                                   n_output_features=3,
                                   n_gaussians=5,
                                   n_neurons=[50, 50, 50, 50, 50, 50]):

    X = tf.placeholder(tf.float32, shape=[None, n_input_features], name='X')
    Y = tf.placeholder(tf.float32, shape=[None, n_output_features], name='Y')

    current_input = X
    for layer_i in range(len(n_neurons)):
        current_input = tfl.linear(
            inputs=current_input,
            num_outputs=n_neurons[layer_i],
            activation_fn=tf.nn.tanh,
            scope='layer/' + str(layer_i))

    means = tf.reshape(
        tfl.linear(
            inputs=current_input,
            num_outputs=n_output_features * n_gaussians,
            activation_fn=None,
            scope='means'), [-1, n_output_features, n_gaussians])
    sigmas = tf.maximum(
        tf.reshape(
            tfl.linear(
                inputs=current_input,
                num_outputs=n_output_features * n_gaussians,
                activation_fn=tf.nn.relu,
                scope='sigmas'), [-1, n_output_features, n_gaussians]), 1e-10)
    weights = tf.reshape(
        tfl.linear(
            inputs=current_input,
            num_outputs=n_output_features * n_gaussians,
            activation_fn=tf.nn.softmax,
            scope='weights'), [-1, n_output_features, n_gaussians])

    Y_3d = tf.reshape(Y, [-1, n_output_features, 1])
    p = gausspdf(Y_3d, means, sigmas)
    weighted = weights * p
    sump = tf.reduce_sum(weighted, axis=2)
    negloglike = -tf.log(tf.maximum(sump, 1e-10))
    cost = tf.reduce_mean(tf.reduce_mean(negloglike, 1))
    return X, Y, cost, means, sigmas, weights


def train_multiple_gaussians_model():
    img1 = imresize(astronaut(), (64, 64))
    img2 = imresize(coffee(), (64, 64))
    xs1, ys1 = get_data(img1)
    xs2, ys2 = get_data(img2)
    xs = np.r_[xs1, xs2]
    ys = np.r_[ys1, ys2]
    n_iterations = 500
    batch_size = 100
    fig, ax = plt.subplots(1, 1)
    with tf.Graph().as_default() as g, tf.Session(graph=g) as sess:
        X, Y, cost, means, sigmas, weights = build_multiple_gaussians_model()
        optimizer = tf.train.AdamOptimizer(0.001).minimize(cost)
        init_op = tf.global_variables_initializer()
        # Here we tell tensorflow that we want to initialize all
        # the variables in the graph so we can use them
        # This will set W and b to their initial random normal value.
        sess.run(init_op)
        # We now run a loop over epochs
        for it_i in range(n_iterations):
            idxs = np.random.permutation(range(len(xs)))
            n_batches = len(idxs) // batch_size
            for batch_i in range(n_batches):
                idxs_i = idxs[batch_i * batch_size:(batch_i + 1) * batch_size]
                sess.run(optimizer, feed_dict={X: xs[idxs_i], Y: ys[idxs_i]})
            this_cost = sess.run([cost], feed_dict={X: xs, Y: ys})
            print('cost:', this_cost)
            if (it_i + 1) % 20 == 0:
                y_mu, y_dev, y_pi = sess.run(
                    [means, sigmas, weights],
                    feed_dict={X: xs[:np.prod(img1.shape[:2])]})
                if False:
                    ys_pred = np.sum(y_mu * y_pi, axis=2)
                    img = np.clip(ys_pred, 0, 1)
                    ax.imshow(img.reshape(img1.shape))
                else:
                    ys_pred = np.array([
                        y_mu[obv, :, idx]
                        for obv, idx in enumerate(np.argmax(y_pi.sum(1), 1))
                    ])
                    img = np.clip(ys_pred.reshape(img1.shape), 0, 1)
                    ax.imshow(img.reshape(img1.shape))
                plt.show()
                fig.canvas.draw()
