import numpy as np
import lumicks.pylake as lk
from copy import deepcopy
from util import load_estimates, extract_estimates, thresholding_algo


def build_handles_model():
    return lk.inverted_odijk('handles') + lk.force_offset('handles')


def build_composite_model():
    comp_wrt_f = lk.odijk('handles') \
        + lk.inverted_marko_siggia_simplified('protein')
    return comp_wrt_f.invert(interpolate=True,
                             independent_min=0,
                             independent_max=90) + lk.force_offset('handles')


def compute_bic(f, d, estimates, model):
    fit = lk.FdFit(model)
    fit.add_data('slice', f, d)
    load_estimates(fit, estimates)
    bic = fit.bic
    fit.fit()
    return (bic, extract_estimates(fit))


def group_continuous(indices):
    groups = [[indices[0]]]
    current_group = 0
    for index in indices[1:]:
        if index == groups[current_group][-1] + 1:
            groups[current_group].append(index)
        else:
            groups.append([index])
            current_group += 1
    return groups


def stupid_interp(data, repicants):
    interp_data = []
    for point in data:
        for replicate in range(repicants):
            interp_data.append(point)
    return np.asarray(interp_data)


def zoom_unfold(f, d, first_index, almost_peak, end_peak, model,
                lag_factor=2, threshold=4., influence=0, verbose=False):
    if verbose:
        print(first_index, almost_peak, end_peak)
    ref_fit = lk.FdFit(deepcopy(model))
    ref_fit.add_data('before',
                     f[first_index:almost_peak], d[first_index:almost_peak])
    ref_fit.fit()
    estimates = extract_estimates(ref_fit)
    bix = []
    for index in range(almost_peak, end_peak):
        (bic, estimates) = \
            compute_bic(f[first_index:index], d[first_index:index], estimates,
                        model)
        bix.append(bic)
    # peaksign = thresholding_algo(np.diff(bix), len(bix) // lag_factor,
    #                              threshold, influence)['signals']

    return np.argmax(np.diff(bix)) + almost_peak


def running_fit(f, d, first_index, last_index, slice_width, model,
                lag_factor=2, threshold=4., influence=0, verbose=False):
    if last_index < 0:
        last_index = len(f) + last_index
    slices = [slice(first_index, i + slice_width)
              for i in range(first_index, last_index, slice_width)]
    fit = lk.FdFit(deepcopy(model))
    fit.add_data('first', f[slices[0]], d[slices[0]])
    estimates = extract_estimates(fit)
    bix = []
    for slc in slices:
        (bic, estimates) = compute_bic(f[slc], d[slc], estimates,
                                       deepcopy(model))
        bix.append(bic)

    if verbose:
        import matplotlib.pyplot as plt
        plt.plot(bix)
        plt.figure()
        plt.plot(np.diff(bix))

    peaksign = thresholding_algo(np.diff(bix), len(bix) // lag_factor,
                                 threshold, influence)['signals']
    if verbose:
        print(peaksign)
    peaksign_interp = stupid_interp(peaksign, slice_width)
    grouped_peak_signals = \
        group_continuous(np.argwhere(peaksign_interp >= 1).flatten()
                         + first_index + slice_width)  # look here and below

    exact_unfolds = []
    for group in grouped_peak_signals:
        exact_unfolds.append(zoom_unfold(f, d, first_index,
                                         group[0] - slice_width,  # look here and above
                                         group[-1], model,
                                         lag_factor, threshold, influence))

    return {'slices': slices, 'bix': bix,
            'signals': peaksign, 'groups': grouped_peak_signals,
            'unfolds': exact_unfolds}


