"""Tools for parsing TIMIT.

TIMIT is described here: https://catalog.ldc.upenn.edu/docs/LDC93S1/

Copyright Parag K. Mital, September 2016.
"""
import os
import numpy as np
from scipy.io import wavfile
from collections import OrderedDict
import tensorflow as tf


def parse_timit_entry(dirpath, file):
    """Summary.

    Parameters
    ----------
    dirpath : TYPE
        Description
    file : TYPE
        Description

    Returns
    -------
    name : TYPE
        Description
    """
    path = os.path.join(dirpath, file)
    phnfile = os.path.join(dirpath, file.strip('.wav') + '.PHN')
    with open(phnfile) as fp:
        phones = []
        for line_i in fp.readlines():
            els = line_i.split(' ')
            phones.append({'start_time': els[0],
                           'end_time': els[1],
                           'phone': els[2].strip()})
    wrdfile = os.path.join(dirpath, file.strip('.wav') + '.WRD')
    with open(wrdfile) as fp:
        words = []
        for line_i in fp.readlines():
            els = line_i.split(' ')
            words.append({'start_time': els[0],
                          'end_time': els[1],
                          'word': els[2].strip()})
    txtfile = os.path.join(dirpath, file.strip('.wav') + '.TXT')
    starttime = 0
    endtime = 0
    with open(txtfile) as fp:
        lines = fp.readlines()
        els = lines[0].split(' ')
        starttime = els[0]
        endtime = els[1]
        text = " ".join(els[2:]).strip()
    entry = {
        'path': path,
        'name': file,
        'phones': phones,
        'words': words,
        'start': starttime,
        'end': endtime,
        'text': text
    }
    return entry


def parse_timit(timit_dir='/home/parag/kadmldb/data/speech/TIMIT/TIMIT'):
    """Summary.

    Returns
    -------
    name : TYPE
        Description

    Parameters
    ----------
    timit_dir : str, optional
        Description
    """
    timit = []
    for dirpath, dirnames, filenames in os.walk(timit_dir):
        for file in filenames:
            if file.endswith('wav'):
                timit.append(parse_timit_entry(dirpath, file))
    phones = list(set([ph['phone']
                       for t in timit for ph in t['phones']]))
    phones.sort()
    phones = phones + ['_']
    words = list(set([ph['word']
                      for t in timit for ph in t['words']]))
    words.sort()
    encoder = OrderedDict(zip(phones, range(len(phones))))
    decoder = OrderedDict(zip(range(len(phones)), phones))
    return {
        'data': timit,
        'phones': phones,
        'words': words,
        'encoder': encoder,
        'decoder': decoder
    }


def preprocess(file):
    """Summary

    Parameters
    ----------
    file : TYPE
        Description

    Returns
    -------
    name : TYPE
        Description
    """
    fft_size = 256
    sr, s = wavfile.read(file)
    s = s / np.abs(s).max()
    mag, phs = dft.forward(s, hop_size=128, fft_size=fft_size)
    cqft = dft.mel(fft_size, sr)
    mel = np.dot(mag, cqft[:fft_size // 2, :])
    mfcc = dft.mfcc(mel, 1, 13).astype(np.float32)
    return mfcc


def create_observation(el, max_sequence_length, hop_size=128):
    """Summary

    Parameters
    ----------
    el : TYPE
        Description
    max_sequence_length : TYPE
        Description
    hop_size : int, optional
        Description

    Returns
    -------
    name : TYPE
        Description
    """
    # grab MFCCs
    all_Xs = preprocess(el['path'])

    # storage for our observation
    X, Y = [], []

    # grab corresponding phones
    for i, ph in enumerate(el['phones']):

        # grad start/end time in samples
        s1, s2 = ph['start_time'], ph['end_time']

        # convert to frames
        f1, f2 = [min(int(np.round(float(s) / hop_size)), len(all_Xs))
                  for s in (s1, s2)]
        n_frames = f2 - f1

        # if we have enough for this observation
        if len(X) + n_frames > max_sequence_length:

            # make sure we have data
            if len(X) and len(Y):
                yield (np.array(X),
                       np.array(Y))
                # reset for next observation
                X, Y = [], []

        # if this observation is smaller than our max sequence length
        if n_frames < max_sequence_length:
            # Grab the data from this phone
            for f in range(f1, f2):
                X.append(all_Xs[f])
            Y.append(ph['phone'])
            Y.append('_')


def batch_generator(timit, batch_size=10, max_sequence_length=50):
    """Summary

    Parameters
    ----------
    timit : TYPE
        Description
    batch_size : int, optional
        Description
    max_sequence_length : int, optional
        Description

    Returns
    -------
    name : TYPE
        Description
    """
    data = timit['data']

    # randomize order
    rand_idxs = np.random.permutation(range(len(data)))

    # number of batchs
    n_batches = len(data) // batch_size

    seq_lens_Xs, seq_lens_Ys = [], []
    Xs, Ys = [], []
    batch_i = 0
    while batch_i < n_batches:
        # grab next random recording
        el = rand_idxs[batch_i]
        batch_i += 1

        for X, Y in create_observation(data[el], max_sequence_length):
            # keep track of the original sequence lengths
            seq_lens_Xs.append(len(X))
            seq_lens_Ys.append(len(Y))

            # encode phones to integers
            Y_enc = np.array([timit['encoder'][y_i] for y_i in Y],
                              dtype=np.int32)[np.newaxis]

            # zero-pad to max sequence length
            if len(X) < max_sequence_length:
                X_pad = np.zeros((1, max_sequence_length, X.shape[-1]))
                X_pad[:, :len(X), :] = X
            else:
                X_pad = X[np.newaxis]

            # append to minibatches
            if len(Xs):
                Xs = np.r_[Xs, X_pad]
                Ys = np.r_[Ys, Y_enc]
            else:
                Xs = X_pad
                Ys = Y_enc

            # we've got enough observations for a minibatch
            if len(Xs) == batch_size:
                yield Xs, Ys, seq_lens_Xs, seq_lens_Ys
                Xs, Ys = [], []
                seq_lens_Xs, seq_lens_Ys = [], []


def sparse_tuple_from(sequences, dtype=np.int32):
    """Summary

    Parameters
    ----------
    sequences : TYPE
        Description
    dtype : TYPE, optional
        Description

    Returns
    -------
    name : TYPE
        Description
    """
    indices = []
    values = []

    for n, seq in enumerate(sequences):
        indices.extend(zip([n] * len(seq), range(len(seq))))
        values.extend(seq)

    indices = np.asarray(indices, dtype=np.int64)
    values = np.asarray(values, dtype=dtype)
    shape = np.asarray([len(sequences),
                        np.asarray(indices).max(0)[1] + 1], dtype=np.int64)

    return indices, np.squeeze(values), shape


def build_model(batch_size=100, sequence_length=50, n_features=13,
                n_cells=100, n_layers=2, n_classes=61, bi=True):
    """Summary

    Parameters
    ----------
    batch_size : int, optional
        Description
    sequence_length : int, optional
        Description
    n_features : int, optional
        Description
    n_cells : int, optional
        Description
    n_layers : int, optional
        Description
    n_classes : int, optional
        Description
    bi : bool, optional
        Description

    Returns
    -------
    name : TYPE
        Description
    """
    with tf.name_scope('input'):
        X = tf.placeholder(
            tf.float32, shape=[None, sequence_length, n_features], name='X')
        seqlens_X = tf.placeholder(tf.int32, shape=[None], name='seqlens')

    with tf.name_scope('rnn'):
        # Create the rnn cells
        forward_cells = tf.nn.rnn_cell.LSTMCell(
            n_cells, use_peepholes=True, state_is_tuple=True)
        if bi:
            backward_cells = tf.nn.rnn_cell.LSTMCell(
                n_cells, use_peepholes=True, state_is_tuple=True)

        # If there are many layers
        if n_layers > 1:
            forward_cells = tf.nn.rnn_cell.MultiRNNCell(
                [forward_cells] * n_layers, state_is_tuple=True)
            if bi:
                backward_cells = tf.nn.rnn_cell.MultiRNNCell(
                    [backward_cells] * n_layers, state_is_tuple=True)

        # Initial state
        initial_state_fw = forward_cells.zero_state(
            tf.shape(X)[0], tf.float32)
        if bi:
            initial_state_bw = backward_cells.zero_state(
                tf.shape(X)[0], tf.float32)

        # Connect it to the input
        if bi:
            outputs, output_states = \
                tf.nn.bidirectional_dynamic_rnn(
                    forward_cells, backward_cells,
                    X, tf.cast(seqlens_X, tf.int64),
                    initial_state_fw, initial_state_bw)
        else:
            outputs, output_states = tf.nn.dynamic_rnn(
                forward_cells, X, seqlens_X, initial_state_fw)

        # Pack into [timesteps, batch_size, 2 * n_cells]
        outputs = tf.pack(outputs)
        # Reshape to  [timesteps * batch_size, 2 * n_cells]
        outputs = tf.reshape(outputs, [-1, 2 * n_cells if bi else n_cells])

    with tf.variable_scope('prediction'):
        W = tf.get_variable(
            "W",
            shape=[2 * n_cells if bi else n_cells, n_classes],
            initializer=tf.random_normal_initializer(stddev=0.1))
        b = tf.get_variable(
            "b",
            shape=[n_classes],
            initializer=tf.random_normal_initializer(stddev=0.1))

        # Find the output prediction of every single character in our minibatch
        # we denote the pre-activation prediction, logits.
        logits = tf.matmul(outputs, W) + b
        logits = tf.reshape(logits, [sequence_length, -1, n_classes])

    with tf.name_scope('output'):
        Y = tf.sparse_placeholder(tf.int32, name='Y')
        seqlens_Y = tf.placeholder(tf.int32, shape=[batch_size], name='seqlen')

    with tf.name_scope('loss'):
        losses = tf.nn.ctc_loss(logits, Y, seqlens_X,
                                preprocess_collapse_repeated=True,
                                ctc_merge_repeated=False)
        cost = tf.reduce_mean(losses)

    with tf.name_scope('decoder'):
        decoder, log_prob = tf.nn.ctc_beam_search_decoder(
            logits, seqlens_Y)
        acc = tf.reduce_mean(tf.edit_distance(tf.cast(
            decoder[0], tf.int32), Y))

    return {'X': X, 'Y': Y, 'cost': cost, 'decoder': decoder, 'acc': acc,
            'seqlens_X': seqlens_X, 'seqlens_Y': seqlens_Y}


def train():
    """Summary

    Returns
    -------
    name : TYPE
        Description
    """
    n_epochs = 10000
    batch_size = 1
    sequence_length = 80
    ckpt_name = 'timit.ckpt'

    timit = parse_timit()
    g = tf.Graph()
    with tf.Session(graph=g) as sess, g.as_default():
        model = build_model(batch_size=batch_size,
                            sequence_length=sequence_length,
                            n_classes=len(timit['encoder']) + 1)
        opt = tf.train.AdamOptimizer().minimize(model['cost'])
        sess.run(tf.initialize_all_variables())
        saver = tf.train.Saver()
        if os.path.exists(ckpt_name):
            saver.restore(sess, ckpt_name)
            print("Model restored.")
        for epoch_i in range(n_epochs):
            avg_acc = 0
            avg_cost = 0
            it_i = 0
            for X, Y, seqlens_X, seqlens_Y in batch_generator(timit, batch_size, sequence_length):
                feed_dict = {
                    model['X']: X,
                    model['Y']: sparse_tuple_from(Y),
                    model['seqlens_X']: seqlens_X,
                    model['seqlens_Y']: seqlens_Y
                }
                this_acc, this_cost, _ = sess.run(
                    [model['acc'], model['cost'], opt], feed_dict=feed_dict)
                it_i += 1
                avg_acc += this_acc
                avg_cost += this_cost
            print(epoch_i, avg_acc / it_i, avg_cost / it_i)
            # Save the variables to disk.
            save_path = saver.save(sess, "./" + ckpt_name,
                                   global_step=epoch_i,
                                   write_meta_graph=True)
            print("Model saved in file: %s" % save_path)


if __name__ == '__main__':
    train()
