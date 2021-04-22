import numpy as np
import matplotlib.pyplot as plt
import h5py
from scipy.stats import hmean, pearsonr
from copy import deepcopy


def find_pulls(signal, bins=100, stepsize=1000, verbose=False):
    bars = plt.hist(signal[::stepsize], bins=bins)
    highest = np.argmax(bars[0])
    second = np.argmax([heigth for index, heigth in enumerate(bars[0])
                        if index != highest])

    low_signal = bars[1][min(highest, second) + 1]
    high_signal = bars[1][max(highest, second)]

    print(bars[1])
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


def smooth(x, kernel_size=1000):
    kernel = np.ones(kernel_size) / kernel_size
    return np.convolve(x, kernel, mode='same')


def baseline(d, f, pulling_start, degree=8):
    poly = np.polyfit(d[:pulling_start],f[:pulling_start],degree)
    P = np.poly1d(poly)
    return f - P(d)


def pearsonerer(x, y):
    return lambda slc: pearsonr(x[slc], y[slc])[0]


def clip_data_dict(data_dict, head=0, tail=False, downsample=1):
    new_dict = {}
    for key, vector in data_dict.items():
        if not tail:
            new_dict[key] = vector[head::downsample]
        else:
            new_dict[key] = vector[head:tail:downsample]
    return new_dict


def find_ditch(f1, f2, box_size=100):
    sf1 = smooth(f1)
    sf2 = smooth(f2)
    pearsoner = pearsonerer(sf1,sf2)
    # def pearsoner(slc):
    #     if not slc.start % 10000:
    #         print(slc.start)
    #     return pearsonr(sf1[slc],sf2[slc])[0]
    # slices = [slice(i, i + box_size)
              # for i in range(len(sf1) - box_size)]
    downsample = 10
    #questionmark = 
    slices = [slice(i * downsample, i * downsample + box_size)
              for i in range(len(sf1) // downsample - box_size // downsample)]
    print(len(sf1))
    print('slc', len(slices))
    pearsons = np.asarray(list(map(pearsoner, slices)))
    smearsons = smooth(pearsons)
                   # abs?
    ditchpoint = np.argmax(np.diff(smearsons[::6000 // downsample])) \
        * 6000  # very sketchy
    plt.figure()
    plt.plot(smearsons)
    plt.figure()
    plt.title('smearsons')
    plt.plot(np.diff(smearsons[::6000 // downsample]))
    plt.figure()
    plt.plot(sf1)
    plt.plot(sf2)
    plt.ylim((min(sf1),max(sf2)))
    plt.axvline(ditchpoint)
    return ditchpoint
    

def clean_data(data_full, signal_threshold=2):
    end_garbage = np.argwhere(data_full['distance'] > signal_threshold)[0][0]
    data_clean = clip_data_dict(data_full, head=end_garbage)
    end_descent = np.argwhere(data_clean['signal'] < signal_threshold)[0][0]
    data_clipped = clip_data_dict(data_clean, head=end_descent)
    print(end_garbage, end_descent)
    print('len', len(data_clean['force']))
    return data_clean, data_clipped, end_descent
    

def calibrate_data(data_clean, descent_end, pulling_start):
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
    print('ditch:',ditch_index,ditched_distance)
    return data_cal


def autosplit(fname):
    """a mess"""
    # import data
    d = h5py.File(fname, 'r')
    data_full = {
        'signal': np.asarray(d['Trap position']['1X']),
        'distance': np.asarray(d['Distance']['Piezo Distance']),
        'force': np.asarray(d['Force HF']['Force 1x']),
        'force_2': np.asarray(d['Force HF']['Force 2x'])
    }
    toseconds = 1e-9  # from ns
    duration = (d['Force LF']['Force 1x'][:][-1][0]
                - d['Force LF']['Force 1x'][:][0][0]) * toseconds
    frequency = len(data_full['force']) / duration
    d.close()

    plt.figure()
    plt.plot(data_full['signal'])
    plt.plot(data_full['distance'])
    plt.figure()

    # clean up data
    data_clean, data_clipped, end_descent = clean_data(data_full)

    pulls = find_pulls(data_clipped['signal'])
    target = 6000  # target amount of datapoints per curve
    pullens = [pull[1] - pull[0] for pull in pulls]
    pull_points = int(hmean(pullens))  # mean of datapoints per curve. hmean to be close to mode

    print('ppp:',pull_points)
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
            {'pull_force': smooth_force[start:pull_stop][::kernel_size],
             'pull_dist': distance[start:pull_stop][::kernel_size],
             'rlx_force': smooth_force[relax_start:stop][::kernel_size],
             'rlx_dist': distance[relax_start:stop][::kernel_size],
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
