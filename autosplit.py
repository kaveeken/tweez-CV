import numpy as np
import matplotlib.pyplot as plt
import h5py
from scipy.stats import hmean


def find_signal_start(signal):
    SIG_CUTOFF = 2
    first = np.argwhere(np.asarray(signal) < SIG_CUTOFF)[0][0]
    top = np.max(signal[first:])
    new_first = np.argwhere(np.asarray(signal) <= top)[0][0]
    #return signal[new_first:]
    return new_first


def find_first_landing(signal):
    BINS = 100
    edge = np.histogram(signal, bins=BINS)[1][1]
    first = np.argwhere(np.asarray(signal) < edge)[0][0]
    return first
    

def baseline(force, distance, region, degree=4):
    x = distance[region]
    y = force[region]
    P = np.poly1d(np.polyfit(x,y,degree))
    new_force = force - P(distance) 

    return new_force
    

def find_pulls(signal, bins=100, stepsize=1000, verbose=False):
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
        if pulling and relaxing:  # ignore first partial pull
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
            pulls.append((start, index * stepsize))
    return pulls



def autosplit(fname):
    """a mess"""
    d = h5py.File(fname, 'r')
    signal = d['Trap position']['1X']
    start = find_signal_start(signal)
    signal = signal[start:]
    first_land = find_first_landing(signal)
    distance = d['Distance']['Piezo Distance'][start:]
    distance = distance - np.amin(distance)
    force_raw = d['Force HF']['Force 1x'][start:]
    force = baseline(force_raw, distance, slice(0, first_land))

    toseconds = 1e-9
    duration = (d['Force LF']['Force 1x'][:][-1][0]
                - d['Force LF']['Force 1x'][:][0][0]) * toseconds
    frequency = len(force) / duration
    d.close()

    pulls = find_pulls(signal)

    target = 6000  # target amount of datapoints per curve
    pullens = [pull[1] - pull[0] for pull in pulls]
    pull_points = int(hmean(pullens))  # mean of datapoints per curve. hmean to be close to mode
    kernel_size = pull_points // target
    kernel = np.ones(kernel_size) / kernel_size
    smooth_force = np.convolve(force, kernel, mode='same')  # magic?

    curves = {}
    for index, pull in enumerate(pulls):
        downsample = 10000
        region = signal[pull[0]:pull[1]][::downsample]
        bars = plt.hist(region, bins=50)  # [0]: counts, [1]: bin lower limits
        first_top = last_top = 0
        time = 0
        if index:  # milliseconds
            time = int((pull[0] - pulls[index - 1][1]) / frequency * 1000)
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
        rest = time

        padding = len(str(len(pulls)))
        identifier = 'curve_' + str(index + 1).zfill(padding)

        # filter out some cases
        if length > 2 * target * kernel_size \
           or length < target * kernel_size / 2 \
           or smooth_force[start] > 30:
            continue

        curves[identifier] = \
            {'pull_force': smooth_force[start:pull_stop][::kernel_size][:-10],
             'pull_dist': distance[start:pull_stop][::kernel_size][:-10],
             'rlx_force': smooth_force[relax_start:stop][::kernel_size][10:],
             'rlx_dist': distance[relax_start:stop][::kernel_size][10:],
             'full_force': smooth_force[start:stop][::kernel_size],
             'full_dist': distance[start:stop][::kernel_size],
             'sign': signal[start:stop][::kernel_size],
             'rest': rest}
    return curves

# print('somethihng')
# curves = autosplit('/home/kris/proj/.data/tweez/yhsp2.h5')
# print(curves.keys())
# with h5py.File('/home/kris/proj/.data/tweez/test.h5', 'w') as f:
#     for curve_id, curve in curves.items():
#         grp = f.create_group(curve_id)
#         grp.attrs.create('rest', data=curve['rest'])
#         for name in ['pull_force', 'pull_dist', 'rlx_force', 'rlx_dist',
#                      'full_force', 'full_dist']:
#             grp.create_dataset(name, data=curve[name])
