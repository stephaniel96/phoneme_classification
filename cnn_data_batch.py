""" Data for training and testing """
from __future__ import division
import random
import glob
import scipy.io as sio
import numpy as np

# The size of our audio sample
INTERVAL_SIZE = 1024
FREQUENCY = 16000
DICT_SIZE = 76

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

def split_data():
    """ Split data so that training and testing do not overlap """
    path = "mat_normalized/*.mat"
    list_files = glob.glob(path)
    length = len(list_files)
    train_files = list_files[:length-100]
    test_files = list_files[length-100:length]

    return train_files, test_files

def next_train_batch(batch_size):
    """ Get next training batch """
    data = np.zeros((batch_size, INTERVAL_SIZE))
    labels = np.zeros((batch_size, DICT_SIZE))
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
        data[i, :] = np.ravel(mat_contents['aud'][start_index:start_index+INTERVAL_SIZE])
        # Find corresponding label
        time = (start_index + INTERVAL_SIZE) / (2* FREQUENCY)
        index = binary_search(mat_contents['intervals'], time)
        key = mat_contents['phonemes'][0, index][0]
        phone_val = m_dict.get(key)
        val_list = np.zeros((1, 76))
        val_list[0, phone_val] = 1
        labels[i, :] = val_list

    return data, labels

def next_test_batch(batch_size):
    """ Get next testing batch """
    data = np.zeros((batch_size, INTERVAL_SIZE))
    labels = np.zeros((batch_size, DICT_SIZE))
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
        data[i, :] = np.ravel(mat_contents['aud'][start_index:start_index+INTERVAL_SIZE])
        # Find corresponding label
        time = (start_index + INTERVAL_SIZE) / (2* FREQUENCY)
        index = binary_search(mat_contents['intervals'], time)
        key = mat_contents['phonemes'][0, index][0]
        phone_val = m_dict.get(key)
        val_list = np.zeros((1, 76))
        val_list[0, phone_val] = 1
        labels[i, :] = val_list

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