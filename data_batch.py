""" Data for training and testing """
from __future__ import division
import random
import glob
import scipy.io as sio
import numpy as np

# The size of our audio sample
INTERVAL_SIZE = 1024
FREQUENCY = 16000

def init_dict():
    """ Dictionary of monophones """
    with open("monophones") as mono_file:
        counter = 0
        c = {}
        for line in mono_file:
            line = line.rstrip('\n')
            c[line] = counter
            counter = counter + 1
    return c

def get_alt_dict():
    """ Alternative dictionary of monophones only 39 """
    alt_dict = {'EH2': 10, 'K': 19,
                'S': 28, 'L': 20, 'M': 21,
                'SH': 29, 'N': 22, 'P': 26,
                'OY0': 25, 'OY2': 25, 'OY1': 25, 'OW2': 24,
                'T': 30, 'OW1': 24, 'EY0': 12, 'EY1': 12, 'EY2': 12,
                'AW2': 4, 'AW1': 4, 'AW0': 4,
                'br': 39, 'cg': 39, 'lg': 39, 'ls': 39, 'ns': 39, 'sil': 39, 'sp': 39,
                'Z': 37, 'W': 35, 'D': 8, 'AH0': 2, 'AH1': 2, 'AH2': 2,
                'B': 6, 'EH1': 10, 'EH0': 10, 'V': 34,
                'IH1': 16, 'IH0': 16, 'IH2': 16,
                'IY2': 17, 'IY1': 17, 'IY0': 17,
                'R': 27, 'AY1': 5, 'ER0': 11,
                'AE1': 1, 'AE2': 1, 'AO1': 3, 'AO2': 3,
                'NG': 23, 'AA0': 0, 'AA2': 0, 'AA1': 0,
                'G': 14, 'TH': 31,
                'F': 13, 'DH': 9, 'HH': 15,
                'UH1': 32, 'UH2': 32, 'UH0': 32,
                'CH': 7, 'UW1': 33, 'UW0': 33, 'UW2': 33,
                'OW0': 24, 'AE0': 1, 'AO0': 3, 'JH': 18,
                'Y': 36, 'ZH': 38, 'AY2': 5, 'ER1': 11,
                'AY0': 5, 'ER2': 11}
    return alt_dict

def replaceZeroes(data):
    min_nonzero = np.min(data[np.nonzero(data)])
    data[data == 0] = min_nonzero
    return data

def split_data():
    """ Split data so that training and testing do not overlap """
    # path = "../../clean_data/mat_normalized/*.mat"
    path = "mat_normalized/*.mat"
    list_files = glob.glob(path)
    length = len(list_files)
    train_files = list_files[:length-100]
    test_files = list_files[length-100:length]

    return train_files, test_files

def next_train_batch(batch_size):
    """ Get next training batch """
    data = np.zeros((batch_size, INTERVAL_SIZE))
    labels = np.zeros(batch_size)
    m_dict = init_dict()
    train, _ = split_data()
    for i in range(batch_size):
        # Randomly chooses a file from the training data
        fname = random.choice(train)
        # Loads the contents of the .mat file
        mat_contents = sio.loadmat(fname)
        # Choose random audio interval from the given .mat file
        aud_length = len(mat_contents['aud'])
        start_index = random.randint(0, aud_length - INTERVAL_SIZE)
        wave_form = mat_contents['aud'][start_index:start_index+INTERVAL_SIZE]
        data[i, :] = np.ravel(wave_form)
        # aud_length = len(mat_contents['aud'])
        # start_index = random.randint(0, aud_length - INTERVAL_SIZE)
        # input_fourier = mat_contents['aud'][start_index:start_index+INTERVAL_SIZE]
        # output_fourier = np.fft.rfft(input_fourier)
        # log_base_10 = np.log10(replaceZeroes(output_fourier))
        # abs_value = np.absolute(log_base_10)
        # data[i, :] = np.ravel(abs_value)
        # Find corresponding label
        time = (start_index + INTERVAL_SIZE) / (2* FREQUENCY)
        index = binary_search(mat_contents['intervals'], time)
        key = mat_contents['phonemes'][0, index][0]
        labels[i] = m_dict.get(key)

    return data, labels

def next_test_batch(batch_size):
    """ Get next testing batch """
    data = np.zeros((batch_size, INTERVAL_SIZE))
    labels = np.zeros(batch_size)
    m_dict = init_dict()
    _, test = split_data()
    for i in range(batch_size):
        # Randomly chooses a file from the testing data
        fname = random.choice(test)
        # Loads the contents of the .mat file
        mat_contents = sio.loadmat(fname)
        # Choose random audio interval from the given .mat file
        aud_length = len(mat_contents['aud'])
        start_index = random.randint(0, aud_length - INTERVAL_SIZE)
        wave_form = mat_contents['aud'][start_index:start_index+INTERVAL_SIZE]
        data[i, :] = np.ravel(wave_form)
        # Find corresponding label
        time = (start_index + INTERVAL_SIZE) / (2* FREQUENCY)
        index = binary_search(mat_contents['intervals'], time)
        key = mat_contents['phonemes'][0, index][0]
        labels[i] = m_dict.get(key)
    return data, labels

def binary_search(intervals, time):
    """ Binary search for the interval index the time falls into """
    length = len(intervals[0])
    lo = 0
    hi = length - 1

    while lo <= hi:
        mid = (lo + hi) // 2
        # print('this is an iteration')
        # print(lo)
        # print(mid)
        # print(hi)
        # print(intervals[0,mid])
        if intervals[0, mid] < time:
            if intervals[1, mid] >= time:
                break
            else:
                lo = mid + 1
        else:
            hi = mid - 1
    return mid