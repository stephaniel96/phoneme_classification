from __future__ import division
import os, random
import glob
import scipy.io as sio
import numpy as np

# The size of our audio sample
INTERVAL_SIZE = 1024
FREQUENCY = 16000

def init_dict():
    with open("monophones") as f:
        counter = 0
        c = {}
        for line in f:
            line = line.rstrip('\n')
            c[line] = counter
            counter = counter + 1
    return c

def next_batch(batch_size):
    path = "../../testfalign/CHAPLIN_MAT/*.mat"
    data = np.zeros((batch_size, INTERVAL_SIZE))
    labels = np.zeros(batch_size)
    d = init_dict()
    for i in range(batch_size):
        # Randomly chooses a file from the training data
        fname = random.choice(glob.glob(path))
        # Loads the contents of the .mat file
        mat_contents = sio.loadmat(fname)
        # Choose random audio interval from the given .mat file
        aud_length = len(mat_contents['aud'])
        start_index = random.randint(0, aud_length - INTERVAL_SIZE)
        data[i,:] = np.ravel(mat_contents['aud'][start_index:start_index+INTERVAL_SIZE])
        # Find corresponding label
        time = (start_index + INTERVAL_SIZE) / (2* FREQUENCY)
        index = binary_search(mat_contents['intervals'], time)
        key = mat_contents['phonemes'][0, index][0]
        labels[i] = d.get(key)

    return data, labels

def binary_search(intervals, time):
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
        if intervals[0,mid] < time:
            if intervals[1,mid] >= time:
                break
            else:
                lo = mid + 1
        else:
            hi = mid - 1
    return mid