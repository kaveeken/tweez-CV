import numpy as np
import matplotlib.pyplot as plt
import h5py
from scipy.stats import hmean, pearsonr
from copy import deepcopy


def find_pulls(signal, bins=100, stepsize=1000, verbose=False):
    """ From a trap position timeseries `signal` identify discrete
    pulls, and return a list of tuples describing the start and end indices of
    each identified pull.

    # Arguments:
    - signal: trap position timeseries array
    - bins: number of bins used in itentifying top and bottom of pulling curves
    - stepsize: degree of downsampling. mostly about performance
    - verbose: whether to track progress by printing every once in a while
    """
    # find the tops and bottoms of pulling curves by binning our signal data
    # this approach may not work with data that includes differently shaped pulls
    # we assume that the peaks and throughs of pulls are consistent and are the most visited states
    #bars = plt.hist(signal[::stepsize], bins=bins)  # doesn't nupy have a binning fn?
    hist, edges = np.histogram(signal[::stepsize], bins=bins)
    highest = np.argmax(hist)
    second = np.argmax([heigth for index, heigth in enumerate(hist)
                        if index != highest])

    low_signal = edges[min(highest, second) + 1]
    high_signal = edges[max(highest, second)]

    pulling = True
    relaxing = True
    pulls = []
    for index, position in enumerate(signal[::stepsize]):
        if not index % 100 and verbose:
            print(index * stepsize)
        if pulling and relaxing:  # ignore first partial pull. assumed: signal starts high and involves an initial approach
            if position < low_signal:
                pulling = False
                relaxing = False
        elif not pulling and not relaxing and position >= low_signal:
            pulling = True
            start = max(index * stepsize - stepsize, 0)
        elif pulling and position > high_signal:
            pulling = False
            relaxing = True
        elif relaxing and position < low_signal:
            pulling = False
            relaxing = False
            pulls.append((start, index * stepsize))  # slice preferable over tuple

    return pulls


def smooth(x, kernel_size=1000):
    """ Smooth a vector `x` using numpy magic I don't fully understand.

    # Arguments:
    - x: vector to be smoothed
    - kernel_size: degree with which to smooth vector `x` (number of points to average)
    """
    kernel = np.ones(kernel_size) / kernel_size
    return np.convolve(x, kernel, mode='same')


def baseline(d, f, pulling_start, degree=8):
    """ Account for bead-bead interactions in force data `f`. In the initial
    approach of two beads (data before `pulling_start` index), fit a polynomial
    funtion to the force:distance relationship, and use that funciton to
    remove bead-bead effects from the `f` data.
    
    # Arguments:
    - d: distance timeseries data
    - f: force timeseries data
    - pulling_start: index of the timeseries data where pulling starts and
    the initial approach has ended.
    - degree: degree of the polynomial to be fitted.
    """
    poly = np.polyfit(d[:pulling_start], f[:pulling_start], degree)
    P = np.poly1d(poly)
    return f - P(d)


def pearsonerer(x, y):
    """ Build a function that takes a slice object and computes pearon
    correlation between timeseries data `x` and `y` at that slice.

    # Arguments:
    - x: first timeseries
    - y: second timeseries
    """
    # idk if this is proper use of lambda
    return lambda slc: pearsonr(x[slc], y[slc])[0]


def clip_data_dict(data_dict, head=0, tail=False, downsample=1):
    """ Take a dictionary containing values of timeseries data (all of the same
    length), and return a new dictionary with subsets of those timeseries.

    # Arguments:
    - data_dict: dictionary containing same-length vectors of data
    - head: number of elements to remove from the start of each vector
    - tail: last element to keep from each vector
    - downsample: degree with with to downsample the data
    """
    new_dict = {}
    for key, vector in data_dict.items():
        if not tail:
            new_dict[key] = vector[head::downsample]
        else:
            new_dict[key] = vector[head:tail:downsample]
    return new_dict


def find_ditch(f1, f2, box_size=100):
    """ Find the point at which two beads touch by looking for large changes
    in local Pearson correlation.

    # Arguments:
    - f1: force timeseries for bead 1
    - f2: force timeseries for bead 2
    - box_size: size of regions to compute local Pearson correlation
    """
    sf1 = smooth(f1)
    sf2 = smooth(f2)
    pearsoner = pearsonerer(sf1, sf2)
    downsample = 10
    slices = [slice(i * downsample, i * downsample + box_size)
              for i in range(len(sf1) // downsample - box_size // downsample)]
    pearsons = np.asarray(list(map(pearsoner, slices)))
    smearsons = smooth(pearsons)  # abs?
    ditchpoint = np.argmax(np.diff(smearsons[::6000 // downsample])) \
        * 6000  # very sketchy
    # plt.figure()
    # plt.plot(smearsons)
    # plt.title('smearsons')
    # plt.figure()
    # plt.plot(np.diff(smearsons[::6000 // downsample]))
    # plt.figure()
    # plt.plot(sf1)
    # plt.plot(sf2)
    # plt.ylim((min(sf1),max(sf2)))
    # plt.axvline(ditchpoint)
    return ditchpoint


def clean_data(data_full, signal_threshold=2):
    """ Remove part of the beginning of a dictionary of timeseries data,
    to where only (part of) the initial approach of beads remains. Returns two
    dictionaries: one with and one without the initial approach, as well as an
    index of where the initial approach ends.

    # Arguments:
    - data_full: dictionary of timeseries OT data (force, distance, signal)
    - signal_threshold: the threshold in the signal data we determine as close
    enough to start working with the data.
    """
    garbage_where = np.argwhere(data_full['distance'] > signal_threshold)
    print(garbage_where)
    if not garbage_where.size:
        end_garbage = 0
    else:
        end_garbage = garbage_where[0][0]
    data_clean = clip_data_dict(data_full, head=end_garbage)
    end_descent = np.argwhere(data_clean['signal'] < signal_threshold)[0][0]
    data_clipped = clip_data_dict(data_clean, head=end_descent)
    print(end_garbage, end_descent)
    print('len', len(data_clean['force']))
    return data_clean, data_clipped, end_descent


def calibrate_data(data_clean, descent_end, pulling_start):
    """ Calibrates and returns a set of data using the `baseline` and 
    `find_ditch` functions.
    """
    based_force_full = baseline(data_clean['distance'], data_clean['force'],
                                pulling_start)
    based_force_2_full = baseline(data_clean['distance'], data_clean['force_2'],
                                  pulling_start)

    ditch_index = find_ditch(based_force_full[descent_end:pulling_start],
                             based_force_2_full[descent_end:pulling_start]) \
                    + descent_end
    ditched_distance = data_clean['distance'][ditch_index]
    data_cal = data_clean
    data_cal['force'] = based_force_full
    data_cal['force_2'] = based_force_2_full
    data_cal['distance'] -= ditched_distance
    print('ditch:', ditch_index, ditched_distance)
    return data_cal


def autosplit(fname):
    """ Reads a C-trap exported file, calibrates the data and splits the data
    up in discrete pulling events. Returns a list of dictionaries, each
    containing key-value pairs for the entire pull-relaxation curve data,
    as well as partial data of the pulling and the relaxation parts.
    Could probably do with some cleaning up.

    # Arguments:
    - fname: filename of a hdp5/lumicks pylake data file. Must include the
    following property keys:
    -- ['Trap position']['1X']
    -- ['Distance']['Piezo Distance']
    -- ['Force HF']['Force 1x']
    -- ['Force HF']['Force 2x']
    """
    # import data
    d = h5py.File(fname, 'r')
    data_full = {
        'signal': np.asarray(d['Trap position']['1X']),
        'distance': np.asarray(d['Distance']['Piezo Distance']),
        'force': np.asarray(d['Force HF']['Force 1x']),
        'force_2': np.asarray(d['Force HF']['Force 2x'])
    }
    # for tracking times (requires 'Force LF' data)
    # toseconds = 1e-9  # from ns
    # duration = (d['Force LF']['Force 1x'][:][-1][0]
    #             - d['Force LF']['Force 1x'][:][0][0]) * toseconds
    # frequency = len(data_full['force']) / duration
    d.close()

    # clean up data
    data_clean, data_clipped, end_descent = clean_data(data_full)

    pulls = find_pulls(data_clipped['signal'])
    target = 6000  # target amount of datapoints per curve
    pullens = [pull[1] - pull[0] for pull in pulls]
    pull_points = int(hmean(pullens))  # mean of datapoints per curve. hmean to be close to mode

    print('ppp:', pull_points)
    print(len(pulls))

    correction = pull_points // 20
    pulling_start = pulls[0][0] + end_descent - correction

    data_cal = calibrate_data(data_clean, end_descent, pulling_start)
    kernel_size = pull_points // target
    data_cal['force'] = smooth(data_cal['force'],
                               kernel_size = kernel_size)

    data_fin = clip_data_dict(data_cal, head=end_descent)

    # bad:
    signal = data_fin['signal']
    distance = data_fin['distance']
    smooth_force = data_fin['force']
    curves = {}
    for index, pull in enumerate(pulls):
        downsample = 10000
        region = signal[pull[0]:pull[1]][::downsample]
        bars = plt.hist(region, bins=50)  # [0]: counts, [1]: bin lower limits
        first_top = last_top = 0
        # time = 0
        #if index:  # milliseconds
            # time = int((pull[0] - pulls[index - 1][1]) / frequency * 1000)
        for jndex, sig in enumerate(region):
            if not first_top and sig > bars[1][-2]:
                first_top = pull[0] + jndex * downsample
            if first_top and not last_top and sig < bars[1][-2]:
                last_top = pull[0] + jndex * downsample
        start = pull[0]
        stop = pull[1]
        length = pull[1] - pull[0]
        pull_stop = first_top
        relax_start = last_top
        # rest = time

        padding = len(str(len(pulls)))
        identifier = 'curve_' + str(index + 1).zfill(padding)

        # filter out some cases
        if length > 2 * target * kernel_size \
           or length < target * kernel_size / 2 \
           or smooth_force[start] > 30:
            continue

        curves[identifier] = \
            {'pull_force': smooth_force[start:pull_stop][::kernel_size],
             'pull_dist': distance[start:pull_stop][::kernel_size],
             'rlx_force': smooth_force[relax_start:stop][::kernel_size],
             'rlx_dist': distance[relax_start:stop][::kernel_size],
             'full_force': smooth_force[start:stop][::kernel_size],
             'full_dist': distance[start:stop][::kernel_size],
             'sign': signal[start:stop][::kernel_size]}
             #'rest': rest}
    return curves
