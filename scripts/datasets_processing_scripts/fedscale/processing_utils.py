import os

import librosa
import numpy as np
import tensorflow as tf

# import soundfile as sf


AUTOTUNE = tf.data.AUTOTUNE
CLASSES = np.array(
    [
        "up",
        "two",
        "sheila",
        "zero",
        "yes",
        "five",
        "one",
        "happy",
        "marvin",
        "no",
        "go",
        "seven",
        "eight",
        "tree",
        "stop",
        "down",
        "forward",
        "learn",
        "house",
        "three",
        "six",
        "backward",
        "dog",
        "cat",
        "wow",
        "left",
        "off",
        "on",
        "four",
        "visual",
        "nine",
        "bird",
        "right",
        "follow",
        "bed",
    ]
)


class HParams(object):
    """Hparams was removed from tf 2.0alpha so this is a placeholder"""

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


hparams = HParams(
    # spectrogramming
    win_length=2048,
    n_fft=2048,
    hop_length=128,
    ref_level_db=50,
    min_level_db=-100,
    # mel scaling
    num_mel_bins=128,
    mel_lower_edge_hertz=0,
    mel_upper_edge_hertz=10000,
    # inversion
    power=1.5,  # for spectral inversion
    griffin_lim_iters=50,
    pad=True,
    #
)


def decode_audio(audio_binary):
    # Decode WAV-encoded audio files to `float32` tensors, normalized
    # to the [-1.0, 1.0] range. Return `float32` audio and a sample rate.
    audio, _ = tf.audio.decode_wav(contents=audio_binary)
    # Since all the data is single channel (mono), drop the `channels`
    # axis from the array.
    return tf.squeeze(audio, axis=-1)


def get_label(file_path):
    parts = tf.strings.split(input=file_path, sep=os.path.sep)
    file_name = parts[-1]
    label = tf.strings.split(input=file_name, sep="_")
    # Note: You'll use indexing here instead of tuple unpacking to enable this
    # to work in a TensorFlow graph.
    return label[0]


def get_waveform_and_label(file_path):
    label = get_label(file_path)
    audio_binary = tf.io.read_file(file_path)
    waveform = decode_audio(audio_binary)
    return waveform, label


def get_spectrogram(waveform):
    # Zero-padding for an audio waveform with less than 16,000 samples.
    input_len = 16000
    waveform = waveform[:input_len]
    zero_padding = tf.zeros([16000] - tf.shape(waveform), dtype=tf.float32)
    # Cast the waveform tensors' dtype to float32.
    waveform = tf.cast(waveform, dtype=tf.float32)
    # Concatenate the waveform with `zero_padding`, which ensures all audio
    # clips are of the same length.
    equal_length = tf.concat([waveform, zero_padding], 0)
    # Convert the waveform to a spectrogram via a STFT.
    spectrogram = tf.signal.stft(equal_length, frame_length=255, frame_step=128)
    # Obtain the magnitude of the STFT.
    spectrogram = tf.abs(spectrogram)
    # Add a `channels` dimension, so that the spectrogram can be used
    # as image-like input data with convolution layers (which expect
    # shape (`batch_size`, `height`, `width`, `channels`).
    spectrogram = spectrogram[..., tf.newaxis]
    return spectrogram


def get_spectrogram_and_label_id(audio, label):
    spectrogram = get_spectrogram(audio)
    label_id = label == CLASSES
    return spectrogram, label_id


def sound_wave_to_mel_spectrogram(sound_wave, sample_rate, spec_h=128, spec_w=128, length=1):
    NUM_MELS = spec_h
    HOP_LENGTH = int(sample_rate * length / (spec_w - 1))
    mel_spec = librosa.feature.melspectrogram(y=sound_wave, sr=sample_rate, hop_length=HOP_LENGTH, n_mels=NUM_MELS)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    return mel_spec_db


def get_mel_spec(waveform, sampling_rate):
    spec_h = 128
    spec_w = 128
    length = 1

    NUM_MELS = spec_h
    n_fft = 2048
    n_mels = 32
    hop_length = 512
    mel_basis = librosa.filters.mel(sampling_rate, 2048, n_mels)
    stft = librosa.stft(waveform.numpy(), n_fft=n_fft, hop_length=hop_length)
    s = np.dot(mel_basis, np.abs(stft) ** 2.0)
    mel_spec_db = librosa.power_to_db(s, ref=np.max)
    # print(mel_spec_db.shape)
    mel_spec_db = np.reshape(mel_spec_db, (32, 32, 1))
    return mel_spec_db


def get_mel_and_label(file_path):
    input_len = 16000
    label = get_label(file_path)
    label_id = label == CLASSES
    # TODO do it on the data as whole
    audio_binary = tf.io.read_file(file_path)
    audio, sampling_rate = tf.audio.decode_wav(contents=audio_binary)
    waveform = tf.squeeze(audio, axis=-1)
    waveform = waveform[:input_len]
    zero_padding = tf.zeros([16000] - tf.shape(waveform), dtype=tf.float32)
    # Cast the waveform tensors' dtype to float32.
    waveform = tf.cast(waveform, dtype=tf.float32)
    # Concatenate the waveform with `zero_padding`, which ensures all audio
    # clips are of the same length.
    equal_length = tf.concat([waveform, zero_padding], 0)
    mel_spec_db_tf = tf.py_function(func=get_mel_spec, inp=[equal_length, sampling_rate], Tout=tf.float32)
    return mel_spec_db_tf, label_id


def preprocess_dataset(files):
    files_ds = tf.data.Dataset.from_tensor_slices(files)
    output_ds = files_ds.map(map_func=get_mel_and_label, num_parallel_calls=AUTOTUNE)

    return output_ds
