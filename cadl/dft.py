"""Utils for performing a DFT using numpy.
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
from scipy.signal import hann
from math import log, exp
from scipy.fftpack import dct
from scipy.special import cbrt


def ztoc_np(re, im):
    """Convert cartesian real, imaginary to polar mag, phs.

    Parameters
    ----------
    re : TYPE
        Description
    im : TYPE
        Description

    Returns
    -------
    name : TYPE
        Description
    """
    return np.sqrt(re**2 + im**2), np.angle(re + im * 1j)


def ctoz_np(mag, phs):
    """Convert polar mag, phs to cartesian real, imaginary.

    Parameters
    ----------
    mag : TYPE
        Description
    phs : TYPE
        Description

    Returns
    -------
    name : TYPE
        Description
    """
    return mag * np.cos(phs), mag * np.sin(phs)


def dft_tf(s):
    """Discrete Fourier Transform.

    Parameters
    ----------
    s : TYPE
        Description

    Returns
    -------
    name : TYPE
        Description
    """
    import tensorflow as tf
    N = s.get_shape().as_list()[-1]
    k = tf.reshape(tf.linspace(0.0, 2 * np.pi / N * (N // 2), N // 2), [1, N // 2])
    x = tf.reshape(tf.linspace(0.0, N - 1, N), [N, 1])
    freqs = tf.matmul(x, k)
    reals = tf.matmul(s, tf.cos(freqs)) * (2.0 / N)
    imags = tf.matmul(s, tf.sin(freqs)) * (2.0 / N)
    return reals, imags


def idft_tf(reals, imags):
    """Inverse Discrete Fourier Transform.

    Parameters
    ----------
    reals : TYPE
        Description
    imags : TYPE
        Description

    Returns
    -------
    name : TYPE
        Description
    """
    import tensorflow as tf
    N = reals.get_shape().as_list()[-1] * 2
    k = tf.reshape(tf.linspace(0.0, 2 * np.pi / N * (N // 2), N // 2), [N // 2, 1])
    x = tf.reshape(tf.linspace(0.0, N - 1, N), [1, N])
    freqs = tf.matmul(k, x)
    return tf.matmul(reals, tf.cos(freqs)) + tf.matmul(imags, tf.sin(freqs))


def stft(signal, hop_size=512, fft_size=2048):
    """Short Time Discrete Fourier Transform w/ Windowing from signal.

    Parameters
    ----------
    signal : TYPE
        Description
    hop_size : int, optional
        Description
    fft_size : int, optional
        Description

    Returns
    -------
    name : TYPE
        Description
    """
    n_hops = len(signal) // hop_size
    s = []
    hann_win = hann(fft_size)
    for hop_i in range(n_hops):
        frame = signal[(hop_i * hop_size):(hop_i * hop_size + fft_size)]
        frame = np.pad(frame, (0, fft_size - len(frame)), 'constant')
        frame *= hann_win
        s.append(frame)
    s = np.array(s)
    N = s.shape[-1]
    k = np.reshape(np.linspace(0.0, 2 * np.pi / N * (N // 2), N // 2), [1, N // 2])
    x = np.reshape(np.linspace(0.0, N - 1, N), [N, 1])
    freqs = np.dot(x, k)
    reals = np.dot(s, np.cos(freqs)) * (2.0 / N)
    imags = np.dot(s, np.sin(freqs)) * (2.0 / N)
    return reals, imags


def istft(re, im, hop_size=512, fft_size=2048):
    """Inverse Short Time Discrete Fourier Transform from real/imag.

    Parameters
    ----------
    re : TYPE
        Description
    im : TYPE
        Description
    hop_size : int, optional
        Description
    fft_size : int, optional
        Description

    Returns
    -------
    name : TYPE
        Description
    """
    N = re.shape[1] * 2
    k = np.reshape(np.linspace(0.0, 2 * np.pi / N * (N // 2), N // 2), [N // 2, 1])
    x = np.reshape(np.linspace(0.0, N - 1, N), [1, N])
    freqs = np.dot(k, x)
    signal = np.zeros((re.shape[0] * hop_size + fft_size,))
    recon = np.dot(re, np.cos(freqs)) + np.dot(im, np.sin(freqs))
    for hop_i, frame in enumerate(recon):
        signal[(hop_i * hop_size): (hop_i * hop_size + fft_size)] += frame
    return signal


def forward(s, **kwargs):
    """Short Time Discrete Fourier Transform w/ Windowing from signal.

    Parameters
    ----------
    s : TYPE
        Description
    **kwargs : TYPE
        Description

    Returns
    -------
    name : TYPE
        Description

    Deleted Parameters
    ------------------
    file : TYPE
        Description
    """
    re, im = stft(s, **kwargs)
    mag, phs = ztoc_np(re, im)
    return mag, phs


def inverse(mag, phs, **kwargs):
    """Inverse Short Time Discrete Fourier Transform from mag/phs.

    Parameters
    ----------
    mag : TYPE
        Description
    phs : TYPE
        Description
    **kwargs : TYPE
        Description

    Returns
    -------
    name : TYPE
        Description

    Deleted Parameters
    ------------------
    file : TYPE
        Description
    """
    re, im = ctoz_np(mag, phs)
    s = istft(re, im, **kwargs)
    return s


def hz2bark(frq):
    """Convert hz to bark scale.

    Parameters
    ----------
    frq : np.ndarray, float
        frequency

    Returns
    -------
    bark : np.ndarray, float
        bark value
    """
    return (26.81 * frq) / (1960 + frq) - 0.51


def bark2hz(bark):
    """Convert bark scale to hz.

    Parameters
    ----------
    bark : np.ndarray, float
        bark value

    Returns
    -------
    frq : np.ndarray, float
        frequency
    """
    return (-19600 * bark - 9996) / (10 * bark - 263)


def hz2pitch(frq, beta=12, ref_frq=440.0):
    """Convert frequency to pitch.

    Parameters
    ----------
    frq : np.ndarray, float
        frequency
    beta : int
        equal divisions of the octave
    ref_frq : float, optional
        Description

    Returns
    -------
    p : np.ndarray, float
        pitch, MIDI value if beta is 12
    """
    return beta * np.log2(frq / ref_frq) + 69


def pitch2hz(p, beta=12, ref_frq=440.0):
    """Convert pitch to frequency.

    Parameters
    ----------
    p : np.ndarray, float
        midi pitch
    beta : int
        equal divisions of the octave
    ref_frq : float, optional
        Description

    Returns
    -------
    frq : np.ndarray, float
        frequency
    """
    return (2**((p - 69) / beta)) * ref_frq


def mel2hz(z, mode='htk'):
    """Convert 'mel scale' frequencies into hz.

    Parameters
    ----------
    z : np.ndarray, float
        'mel scale' frequency
    mode : string
        'htk' uses the mel axis defined in the htkbook
        'slaney' uses slaney's formula

    Returns
    -------
    f : np.ndarray, float
        frequency

    Raises
    ------
    ValueError
        Description
    """
    if mode == 'htk':
        f = 700 * (10**(z / 2595) - 1)

    elif mode == 'slaney':
        f_0 = 0
        f_sp = 200 / 3
        brkfrq = 1000

        # starting mel value for log region
        brkpt = (brkfrq - f_0) / f_sp

        # the magic 1.0711703 which is the ratio
        # needed to get from 1000 hz to 6400 hz in 27 steps, and is
        # *almost* the ratio between 1000 hz and the preceding linear
        # filter center at 933.33333 hz (actually 1000/933.33333 =
        # 1.07142857142857 and  exp(log(6.4)/27) = 1.07117028749447)
        logstep = exp(log(6.4) / 27)

        linpts = (z < brkpt)

        if not np.isscalar(z):
            f = 0 * z
            # fill in parts separately
            f[linpts] = f_0 + f_sp * z[linpts]
            f[~linpts] = brkfrq * np.exp(log(logstep) * (z[~linpts] - brkpt))

        elif linpts:
            f = f_0 + f_sp * z
        else:
            f = brkfrq * exp(log(logstep) * (z - brkpt))

    else:
        raise ValueError("{0} is not a valid mode".format(mode))

    return f


def hz2mel(f, mode='htk'):
    """Convert hz to 'mel scale' frequencies.

    Parameters
    ----------
    f : np.ndarray, float
        frequency
    mode : string
        'htk' uses the mel axis defined in the htkbook
        'slaney' uses slaney's formula

    Returns
    -------
    z : np.ndarray, float
        'mel scale' frequency

    Raises
    ------
    ValueError
        Description
    """
    if mode == 'htk':
        z = 2595 * np.log10(1 + f / 700)
        if np.isscalar(z) and not isinstance(f, np.generic):
            z = float(z)

    elif mode == 'slaney':
        # 133.33333
        f_0 = 0
        # 66.66667
        f_sp = 200 / 3
        brkfrq = 1000

        # starting mel value for log region
        brkpt = (brkfrq - f_0) / f_sp

        # the magic 1.0711703 which is the ratio
        # needed to get from 1000 hz to 6400 hz in 27 steps, and is
        # *almost* the ratio between 1000 hz and the preceding linear
        # filter center at 933.33333 hz (actually 1000/933.33333 =
        # 1.07142857142857 and  exp(log(6.4)/27) = 1.07117028749447)
        logstep = exp(log(6.4) / 27)

        linpts = (f < brkfrq)

        if not np.isscalar(f):
            z = 0 * f
            # fill in parts separately
            z[linpts] = (f[linpts] - f_0) / f_sp
            z[~linpts] = brkpt + np.log(f[~linpts] / brkfrq) / log(logstep)

        elif linpts:
            z = (f - f_0) / f_sp
        else:
            z = brkpt + log(f / brkfrq) / log(logstep)

    else:
        raise ValueError("{0} is not a valid mode".format(mode))

    return z


def weights(binfrqs, N, fs, width=1.0, constamp=False):
    """Calculate a transformation array given the center frequencies.

    Parameters
    ----------
    binfrqs : np.ndarray
        center bin frequencies
    N : int
        fft-size
    fs : int
        sampling rate
    width : float
        width of the overlapping windows, default 1.0
    constamp : bool
        makes integration windows sum to 1.0, not peak at 1.0, default to False

    Returns
    -------
    wts : np.ndarray
        transformation matrix
    """
    # center freqs of each FFT bin
    fftfrqs = np.arange(N // 2 + 1) / N * fs

    # number of filters
    nfilts = len(binfrqs) - 2

    # weights for transformation
    wts = np.zeros((nfilts, N))

    for i in range(nfilts):
        frqs = binfrqs[i:i + 3]

        # scale by width
        frqs = frqs[1] + width * (frqs - frqs[1])

        # lower and upper slopes for all bins
        loslope = (fftfrqs - frqs[0]) / (frqs[1] - frqs[0])
        hislope = (frqs[2] - fftfrqs) / (frqs[2] - frqs[1])

        # intersect them with each other and zero
        wts[i, :N // 2 + 1] = np.maximum(0, np.minimum(loslope, hislope))

    # constant amplitude
    if constamp:
        wts = np.dot(np.diag(2 / (binfrqs[2:2 + nfilts] - binfrqs[:nfilts])), wts)

    return wts


def constantQ(N, fs, width=1.0, minpitch=45, maxpitch=117,
              octave=12, beta=None, ref_frq=440.0, constamp=False):
    """Create a constantQ weight matrix.

    Parameters
    ----------
    N : int
        fft-size
    fs : int
        rate of sampling
    width : float, optional
        width of triangular overlapping windows, default 1.0
    minpitch : int, optional
        minimum pitch of constantQ, default 45
    maxpitch : int, optional
        maximum pitch of constantQ, default 117
    octave : int, optional
        number of pitches per octave,
        also can be considered the EDO or TET, default 12
    beta : int, optional
        number of divisions per octave, default octave * 3
    ref_frq : float, optional
        Description
    constamp : boolean, optional
        constant amplitude across bins, default False

    Returns
    -------
    wts : np.ndarray
        weight matrix

    Deleted Parameters
    ------------------
    ref_freq : float, optional
        frequency of A4, default 440.0

    Raises
    ------
    ValueError
        Description
    """
    if beta is None:
        beta = octave * 3

    if beta % octave != 0:
        raise ValueError("Wrap must be an equal division of beta.")

    # center freqs of each FFT bin
    fftfrqs = np.arange(N // 2 + 1) / N * fs
    nfilts = int(maxpitch - minpitch) * int(beta / octave)

    # calculates pitch values, adjusts for
    low = hz2pitch(pitch2hz(minpitch - octave / beta), beta, ref_frq)
    high = hz2pitch(pitch2hz(maxpitch - octave / beta), beta, ref_frq)

    binfrqs = pitch2hz(np.linspace(low, high, nfilts + 2),
                       beta=beta, ref_frq=ref_frq)

    # weights for transformation
    wts = np.zeros((nfilts, N))

    for i in range(nfilts):
        frqs = binfrqs[i:i + 3]

        # scale by width
        frqs = frqs[1] + width * (frqs - frqs[1])

        # lower and upper slopes for all bins
        loslope = (fftfrqs - frqs[0]) / (frqs[1] - frqs[0])
        hislope = (frqs[2] - fftfrqs) / (frqs[2] - frqs[1])

        # intersect them with each other and zero
        wts[i, :N // 2 + 1] = np.maximum(0, np.minimum(loslope, hislope))

    # constant amplitude
    if constamp:
        wts = np.dot(np.diag(2 / (binfrqs[2:2 + nfilts] - binfrqs[:nfilts])), wts)

    return wts, binfrqs


def mel(N, fs, nfilts=24, width=1.0, minfrq=0, maxfrq=None,
        constamp=False, mode='htk'):
    """Calculate a mel-band transformation matrix.

    Parameters
    ----------
    N : int
        fft-size
    fs : int
        sampling rate
    nfilts : int
        number of mel bands
    width : float
        constant width of each mel band
    minfrq : int
        frequency of the lowest band edge
    maxfrq : int
        frequency of the highest band edge, default at nyquist
    constamp : bool
        makes integration windows sum to 1.0, not peak at 1.0, default to False
    mode : string
        'htk' uses the mel axis defined in the HTKBook
        'slaney' uses Slaney's formula
        if using 'slaney', recommended set constamp to True

    Returns
    -------
    wts : np.ndarray
        transformation matrix

    References
    ----------
    .. [1] Taken from the MATLAB implementation of Dan Ellis' automatic gain control
        See: `documentation <http://labrosa.ee.columbia.edu/matlab/tf_agc/>`_ and
        original MATLAB code. Pythonized 2014-05-23 by Chad Wagner chad@kadenze.com
    """
    # if maxfrq not provided, calculates based on nyquist
    if maxfrq is None:
        maxfrq = fs // 2

    # finds min and max in mel
    minmel = hz2mel(minfrq, mode)
    maxmel = hz2mel(maxfrq, mode)

    # 'center freqs' of mel bands, uniformly spaced between limits
    binfrqs = mel2hz(minmel + np.arange(nfilts + 2) /
                     (nfilts + 1) * (maxmel - minmel), mode)

    # transformation matrix
    wts = weights(binfrqs, N, fs, width, constamp).T

    return wts


def mfcc(X, low=2, high=13, A=1.0, C=1.0, mode="log"):
    """Returns the mel frequency cepstrum coefficients from a mel matrix

    Parameters
    ----------
    X : np.ndarray
        mel matrix of an stft
    low : int
        lowest band to return, default 2
    high : int
        highest band to return, default 13
    A : float
        addition to mel matrix before log transform
    C : float
        multiplication to mel matrix before log transform

    Returns
    -------
    mfcc : np.ndarray
        mel frequency cepstrum coefficients
    """

    if mode == "log":
        # logs of the powers of the mel frequencies
        X = np.log(X * C + A)

    elif mode == "cbrt":
        # cube root the powers of the mel frequencies
        X = cbrt(X)

    else:
        raise ValueError("{0} is not a valid mode".format(mode))

    # discrete cosine transform of each column

    mfcc = dct(X.T, type=2, axis=0, norm='ortho')[max(low - 1, 0):high, :].T

    return mfcc
