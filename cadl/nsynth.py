"""NSynth: WaveNet Autoencoder.
"""
"""
NSynth model code and utilities are licensed under APL from the

Google Magenta project
----------------------
https://github.com/tensorflow/magenta/blob/master/magenta/models/nsynth

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
import tensorflow as tf
from scipy.io import wavfile
import numpy as np
from cadl.utils import download_and_extract_tar
from magenta.models.nsynth import utils
from magenta.models.nsynth import reader
from magenta.models.nsynth.wavenet import masked
import os


def get_model():
    """Summary
    """
    download_and_extract_tar(
        'http://download.magenta.tensorflow.org/models/nsynth/wavenet-ckpt.tar')


def causal_linear(x, n_inputs, n_outputs, name, filter_length, rate,
                  batch_size):
    """Summary

    Parameters
    ----------
    x : TYPE
        Description
    n_inputs : TYPE
        Description
    n_outputs : TYPE
        Description
    name : TYPE
        Description
    filter_length : TYPE
        Description
    rate : TYPE
        Description
    batch_size : TYPE
        Description

    Returns
    -------
    TYPE
        Description
    """
    # create queue
    q_1 = tf.FIFOQueue(rate, dtypes=tf.float32, shapes=(batch_size, n_inputs))
    q_2 = tf.FIFOQueue(rate, dtypes=tf.float32, shapes=(batch_size, n_inputs))
    init_1 = q_1.enqueue_many(tf.zeros((rate, batch_size, n_inputs)))
    init_2 = q_2.enqueue_many(tf.zeros((rate, batch_size, n_inputs)))
    state_1 = q_1.dequeue()
    push_1 = q_1.enqueue(x)
    state_2 = q_2.dequeue()
    push_2 = q_2.enqueue(state_1)

    # get pretrained weights
    W = tf.get_variable(
        name=name + '/W',
        shape=[1, filter_length, n_inputs, n_outputs],
        dtype=tf.float32)
    b = tf.get_variable(
        name=name + '/biases', shape=[n_outputs], dtype=tf.float32)
    W_q_2 = tf.slice(W, [0, 0, 0, 0], [-1, 1, -1, -1])
    W_q_1 = tf.slice(W, [0, 1, 0, 0], [-1, 1, -1, -1])
    W_x = tf.slice(W, [0, 2, 0, 0], [-1, 1, -1, -1])

    # perform op w/ cached states
    y = tf.expand_dims(
        tf.nn.bias_add(
            tf.matmul(state_2, W_q_2[0][0]) + tf.matmul(state_1, W_q_1[0][0]) +
            tf.matmul(x, W_x[0][0]), b), 0)
    return y, (init_1, init_2), (push_1, push_2)


def linear(x, n_inputs, n_outputs, name):
    """Summary

    Parameters
    ----------
    x : TYPE
        Description
    n_inputs : TYPE
        Description
    n_outputs : TYPE
        Description
    name : TYPE
        Description

    Returns
    -------
    TYPE
        Description
    """
    W = tf.get_variable(
        name=name + '/W', shape=[1, 1, n_inputs, n_outputs], dtype=tf.float32)
    b = tf.get_variable(
        name=name + '/biases', shape=[n_outputs], dtype=tf.float32)
    return tf.expand_dims(tf.nn.bias_add(tf.matmul(x[0], W[0][0]), b), 0)


class FastGenerationConfig(object):
    """Configuration object that helps manage the graph."""

    def __init__(self, batch_size=1):
        """."""
        self.batch_size = batch_size

    def build(self, inputs):
        """Build the graph for this configuration.
        Args:
            inputs: A dict of inputs. For training, should contain 'wav'.
        Returns:
            A dict of outputs that includes the 'predictions',
            'init_ops', the 'push_ops', and the 'quantized_input'.
        """
        num_stages = 10
        num_layers = 30
        filter_length = 3
        width = 512
        skip_width = 256
        num_z = 16

        # Encode the source with 8-bit Mu-Law.
        x = inputs['wav']
        batch_size = self.batch_size
        x_quantized = utils.mu_law(x)
        x_scaled = tf.cast(x_quantized, tf.float32) / 128.0
        x_scaled = tf.expand_dims(x_scaled, 2)

        encoding = tf.placeholder(
                name='encoding', shape=[batch_size, num_z], dtype=tf.float32)
        en = tf.expand_dims(encoding, 1)

        init_ops, push_ops = [], []

        ###
        # The WaveNet Decoder.
        ###
        l = x_scaled
        l, inits, pushs = utils.causal_linear(
                x=l,
                n_inputs=1,
                n_outputs=width,
                name='startconv',
                rate=1,
                batch_size=batch_size,
                filter_length=filter_length)

        for init in inits:
            init_ops.append(init)
        for push in pushs:
            push_ops.append(push)

        # Set up skip connections.
        s = utils.linear(l, width, skip_width, name='skip_start')

        # Residual blocks with skip connections.
        for i in range(num_layers):
            dilation = 2**(i % num_stages)

            # dilated masked cnn
            d, inits, pushs = utils.causal_linear(
                    x=l,
                    n_inputs=width,
                    n_outputs=width * 2,
                    name='dilatedconv_%d' % (i + 1),
                    rate=dilation,
                    batch_size=batch_size,
                    filter_length=filter_length)

            for init in inits:
                init_ops.append(init)
            for push in pushs:
                push_ops.append(push)

            # local conditioning
            d += utils.linear(en, num_z, width * 2, name='cond_map_%d' % (i + 1))

            # gated cnn
            assert d.get_shape().as_list()[2] % 2 == 0
            m = d.get_shape().as_list()[2] // 2
            d = tf.sigmoid(d[:, :, :m]) * tf.tanh(d[:, :, m:])

            # residuals
            l += utils.linear(d, width, width, name='res_%d' % (i + 1))

            # skips
            s += utils.linear(d, width, skip_width, name='skip_%d' % (i + 1))

        s = tf.nn.relu(s)
        s = (utils.linear(s, skip_width, skip_width, name='out1') + utils.linear(
                en, num_z, skip_width, name='cond_map_out1'))
        s = tf.nn.relu(s)

        ###
        # Compute the logits and get the loss.
        ###
        logits = utils.linear(s, skip_width, 256, name='logits')
        logits = tf.reshape(logits, [-1, 256])
        probs = tf.nn.softmax(logits, name='softmax')

        return {
                'init_ops': init_ops,
                'push_ops': push_ops,
                'predictions': probs,
                'encoding': encoding,
                'quantized_input': x_quantized,
        }


class Config(object):
    """Configuration object that helps manage the graph."""

    def __init__(self, train_path=None):
        self.num_iters = 200000
        self.learning_rate_schedule = {
                0: 2e-4,
                90000: 4e-4 / 3,
                120000: 6e-5,
                150000: 4e-5,
                180000: 2e-5,
                210000: 6e-6,
                240000: 2e-6,
        }
        self.ae_hop_length = 512
        self.ae_bottleneck_width = 16
        self.train_path = train_path

    def get_batch(self, batch_size):
        assert self.train_path is not None
        data_train = reader.NSynthDataset(self.train_path, is_training=True)
        return data_train.get_wavenet_batch(batch_size, length=6144)

    @staticmethod
    def _condition(x, encoding):
        """Condition the input on the encoding.
        Args:
            x: The [mb, length, channels] float tensor input.
            encoding: The [mb, encoding_length, channels] float tensor encoding.
        Returns:
            The output after broadcasting the encoding to x's shape and adding them.
        """
        mb, length, channels = x.get_shape().as_list()
        enc_mb, enc_length, enc_channels = encoding.get_shape().as_list()
        assert enc_mb == mb
        assert enc_channels == channels

        encoding = tf.reshape(encoding, [mb, enc_length, 1, channels])
        x = tf.reshape(x, [mb, enc_length, -1, channels])
        x += encoding
        x = tf.reshape(x, [mb, length, channels])
        x.set_shape([mb, length, channels])
        return x

    def build(self, inputs, is_training):
        """Build the graph for this configuration.
        Args:
            inputs: A dict of inputs. For training, should contain 'wav'.
            is_training: Whether we are training or not. Not used in this config.
        Returns:
            A dict of outputs that includes the 'predictions', 'loss', the 'encoding',
            the 'quantized_input', and whatever metrics we want to track for eval.
        """
        del is_training
        num_stages = 10
        num_layers = 30
        filter_length = 3
        width = 512
        skip_width = 256
        ae_num_stages = 10
        ae_num_layers = 30
        ae_filter_length = 3
        ae_width = 128

        # Encode the source with 8-bit Mu-Law.
        x = inputs['wav']
        x_quantized = utils.mu_law(x)
        x_scaled = tf.cast(x_quantized, tf.float32) / 128.0
        x_scaled = tf.expand_dims(x_scaled, 2)

        ###
        # The Non-Causal Temporal Encoder.
        ###
        en = masked.conv1d(
                x_scaled,
                causal=False,
                num_filters=ae_width,
                filter_length=ae_filter_length,
                name='ae_startconv')

        for num_layer in range(ae_num_layers):
            dilation = 2**(num_layer % ae_num_stages)
            d = tf.nn.relu(en)
            d = masked.conv1d(
                    d,
                    causal=False,
                    num_filters=ae_width,
                    filter_length=ae_filter_length,
                    dilation=dilation,
                    name='ae_dilatedconv_%d' % (num_layer + 1))
            d = tf.nn.relu(d)
            en += masked.conv1d(
                    d,
                    num_filters=ae_width,
                    filter_length=1,
                    name='ae_res_%d' % (num_layer + 1))

        en = masked.conv1d(
                en,
                num_filters=self.ae_bottleneck_width,
                filter_length=1,
                name='ae_bottleneck')
        en = masked.pool1d(en, self.ae_hop_length, name='ae_pool', mode='avg')
        encoding = en

        ###
        # The WaveNet Decoder.
        ###
        l = masked.shift_right(x_scaled)
        l = masked.conv1d(
                l, num_filters=width, filter_length=filter_length, name='startconv')

        # Set up skip connections.
        s = masked.conv1d(
                l, num_filters=skip_width, filter_length=1, name='skip_start')

        # Residual blocks with skip connections.
        for i in range(num_layers):
            dilation = 2**(i % num_stages)
            d = masked.conv1d(
                    l,
                    num_filters=2 * width,
                    filter_length=filter_length,
                    dilation=dilation,
                    name='dilatedconv_%d' % (i + 1))
            d = self._condition(d,
                                                    masked.conv1d(
                                                            en,
                                                            num_filters=2 * width,
                                                            filter_length=1,
                                                            name='cond_map_%d' % (i + 1)))

            assert d.get_shape().as_list()[2] % 2 == 0
            m = d.get_shape().as_list()[2] // 2
            d_sigmoid = tf.sigmoid(d[:, :, :m])
            d_tanh = tf.tanh(d[:, :, m:])
            d = d_sigmoid * d_tanh

            l += masked.conv1d(
                    d, num_filters=width, filter_length=1, name='res_%d' % (i + 1))
            s += masked.conv1d(
                    d, num_filters=skip_width, filter_length=1, name='skip_%d' % (i + 1))

        s = tf.nn.relu(s)
        s = masked.conv1d(s, num_filters=skip_width, filter_length=1, name='out1')
        s = self._condition(s,
                                                masked.conv1d(
                                                        en,
                                                        num_filters=skip_width,
                                                        filter_length=1,
                                                        name='cond_map_out1'))
        s = tf.nn.relu(s)

        ###
        # Compute the logits and get the loss.
        ###
        logits = masked.conv1d(s, num_filters=256, filter_length=1, name='logits')
        logits = tf.reshape(logits, [-1, 256])
        probs = tf.nn.softmax(logits, name='softmax')
        x_indices = tf.cast(tf.reshape(x_quantized, [-1]), tf.int32) + 128
        loss = tf.reduce_mean(
                tf.nn.sparse_softmax_cross_entropy_with_logits(
                        logits=logits, labels=x_indices, name='nll'),
                0,
                name='loss')

        return {
                'predictions': probs,
                'loss': loss,
                'eval': {
                        'nll': loss
                },
                'quantized_input': x_quantized,
                'encoding': encoding,
        }


def inv_mu_law(x, mu=255.0):
    """A TF implementation of inverse Mu-Law.

    Parameters
    ----------
    x
        The Mu-Law samples to decode.
    mu
        The Mu we used to encode these samples.

    Returns
    -------
    out
        The decoded data.
    """
    x = np.array(x).astype(np.float32)
    out = (x + 0.5) * 2. / (mu + 1)
    out = np.sign(out) / mu * ((1 + mu)**np.abs(out) - 1)
    out = np.where(np.equal(x, 0), x, out)
    return out


def load_audio(wav_file, sample_length=64000):
    """Summary

    Parameters
    ----------
    wav_file : TYPE
        Description
    sample_length : int, optional
        Description

    Returns
    -------
    TYPE
        Description
    """
    wav_data = np.array([utils.load_audio(wav_file)[:sample_length]])
    wav_data_padded = np.zeros((1, sample_length))
    wav_data_padded[0, :wav_data.shape[1]] = wav_data
    wav_data = wav_data_padded
    return wav_data


def load_nsynth(batch_size=1, sample_length=64000):
    """Summary

    Parameters
    ----------
    encoding : bool, optional
        Description
    batch_size : int, optional
        Description
    sample_length : int, optional
        Description

    Returns
    -------
    TYPE
        Description
    """
    config = Config()
    with tf.device('/gpu:0'):
        X = tf.placeholder(tf.float32, shape=[batch_size, sample_length])
        graph = config.build({"wav": X}, is_training=False)
        graph.update({'X': X})
    return graph


def load_fastgen_nsynth(batch_size=1, sample_length=64000):
    """Summary

    Parameters
    ----------
    batch_size : int, optional
        Description
    sample_length : int, optional
        Description

    Returns
    -------
    TYPE
        Description
    """
    config = FastGenerationConfig(batch_size)
    X = tf.placeholder(tf.float32, shape=[batch_size, 1])
    graph = config.build({"wav": X})
    graph.update({'X': X})
    return graph


def sample_categorical(pmf):
    """Sample from a categorical distribution.
    Args:
        pmf: Probablity mass function. Output of a softmax over categories.
            Array of shape [batch_size, number of categories]. Rows sum to 1.
    Returns:
        idxs: Array of size [batch_size, 1]. Integer of category sampled.
    """
    if pmf.ndim == 1:
        pmf = np.expand_dims(pmf, 0)
    batch_size = pmf.shape[0]
    cdf = np.cumsum(pmf, axis=1)
    rand_vals = np.random.rand(batch_size)
    idxs = np.zeros([batch_size, 1])
    for i in range(batch_size):
        idxs[i] = cdf[i].searchsorted(rand_vals[i])
    return idxs


def load_batch(files, sample_length=64000):
    """Load a batch of data from either .wav or .npy files.
    Args:
        files: A list of filepaths to .wav or .npy files
        sample_length: Maximum sample length
    Returns:
        batch_data: A padded array of audio or embeddings [batch, length, (dims)]
    """
    batch_data = []
    max_length = 0
    is_npy = (os.path.splitext(files[0])[1] == ".npy")
    # Load the data
    for f in files:
        if is_npy:
            data = np.load(f)
            batch_data.append(data)
        else:
            data = utils.load_audio(f, sample_length, sr=16000)
            batch_data.append(data)
        if data.shape[0] > max_length:
            max_length = data.shape[0]
    # Add padding
    for i, data in enumerate(batch_data):
        if data.shape[0] < max_length:
            if is_npy:
                padded = np.zeros([max_length, +data.shape[1]])
                padded[:data.shape[0], :] = data
            else:
                padded = np.zeros([max_length])
                padded[:data.shape[0]] = data
            batch_data[i] = padded
    # Return arrays
    batch_data = np.array(batch_data)
    return batch_data


def save_batch(batch_audio, batch_save_paths):
    for audio, name in zip(batch_audio, batch_save_paths):
        tf.logging.info("Saving: %s" % name)
        wavfile.write(name, 16000, audio)


def encode(wav_data, checkpoint_path, sample_length=64000):
    """Generate an array of embeddings from an array of audio.
    Args:
        wav_data: Numpy array [batch_size, sample_length]
        checkpoint_path: Location of the pretrained model.
        sample_length: The total length of the final wave file, padded with 0s.
    Returns:
        encoding: a [mb, 125, 16] encoding (for 64000 sample audio file).
    """
    if wav_data.ndim == 1:
        wav_data = np.expand_dims(wav_data, 0)
        batch_size = 1
    elif wav_data.ndim == 2:
        batch_size = wav_data.shape[0]

    # Load up the model for encoding and find the encoding of "wav_data"
    session_config = tf.ConfigProto(allow_soft_placement=True)
    with tf.Graph().as_default(), tf.Session(config=session_config) as sess:
        hop_length = Config().ae_hop_length
        wav_data, sample_length = utils.trim_for_encoding(
            wav_data, sample_length, hop_length)
        net = load_nsynth(batch_size=batch_size, sample_length=sample_length)
        saver = tf.train.Saver()
        saver.restore(sess, checkpoint_path)
        encodings = sess.run(net["encoding"], feed_dict={net["X"]: wav_data})
    return encodings


def synthesize(encodings,
               save_paths,
               hop_length=None,
               checkpoint_path="model.ckpt-200000",
               samples_per_save=1000):
    """Synthesize audio from an array of embeddings.
    Args:
        encodings: Numpy array with shape [batch_size, time, dim].
        save_paths: Iterable of output file names.
        checkpoint_path: Location of the pretrained model. [model.ckpt-200000]
        samples_per_save: Save files after every amount of generated samples.
    """
    if hop_length is None:
        hop_length = Config().ae_hop_length
    # Get lengths
    batch_size = encodings.shape[0]
    encoding_length = encodings.shape[1]
    total_length = encoding_length * hop_length

    session_config = tf.ConfigProto(allow_soft_placement=True)
    with tf.Graph().as_default(), tf.Session(config=session_config) as sess:
        net = load_fastgen_nsynth(batch_size=batch_size)
        saver = tf.train.Saver()
        saver.restore(sess, checkpoint_path)

        # initialize queues w/ 0s
        sess.run(net["init_ops"])

        # Regenerate the audio file sample by sample
        audio_batch = np.zeros((batch_size, total_length,), dtype=np.float32)
        audio = np.zeros([batch_size, 1])

        for sample_i in range(total_length):
            enc_i = sample_i // hop_length
            pmf = sess.run(
                [net["predictions"], net["push_ops"]],
                feed_dict={
                    net["X"]: audio,
                    net["encoding"]: encodings[:, enc_i, :]
                })[0]
            sample_bin = sample_categorical(pmf)
            audio = utils.inv_mu_law_numpy(sample_bin - 128)
            audio_batch[:, sample_i] = audio[:, 0]
            if sample_i % 100 == 0:
                tf.logging.info("Sample: %d" % sample_i)
            if sample_i % samples_per_save == 0:
                save_batch(audio_batch, save_paths)
    save_batch(audio_batch, save_paths)
