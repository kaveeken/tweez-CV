import numpy as np


def load_estimates(fit, est):
    for param_key, param in est.items():
        for field_key in param.keys():
            if field_key == 'value':
                fit[param_key].value = param["value"]
            elif field_key == 'upper_bound':
                fit[param_key].upper_bound = param["upper_bound"]
            elif field_key == 'lower_bound':
                fit[param_key].lower_bound = param["lower_bound"]
            elif field_key == 'fixed':
                fit[param_key].fixed = param["fixed"]


def extract_estimates(fit):
    est = {}
    for key in fit.params.keys:
        est[key] = {"value": fit.params[key].value,
                    "upper_bound": fit.params[key].upper_bound,
                    "lower_bound": fit.params[key].lower_bound,
                    "fixed": fit.params[key].fixed}
    return est


def average_around(data, index, half_n=10):
    # this needs some sanity checks around the indexing
    subset = data[index - half_n: index + half_n]
    return {"mean": np.mean(subset), "std": np.std(subset)}


def thresholding_algo(y, lag, threshold, influence):
    """Implementation of the noise-resistant peak-finding algorithm cited below.

    Reference:
    Brakel, J.P.G. van (2014). "Robust peak detection algorithm using z-scores". Stack Overflow. Available at: https://stackoverflow.com/questions/22583391/peak-signal-detection-in-realtime-timeseries-data/22640362#22640362 (version: 2020-11-08).
    """
    signals = np.zeros(len(y))
    filteredY = np.array(y)
    avgFilter = [0]*len(y)
    stdFilter = [0]*len(y)
    avgFilter[lag - 1] = np.mean(y[0:lag])
    stdFilter[lag - 1] = np.std(y[0:lag])
    for i in range(lag, len(y)):
        if abs(y[i] - avgFilter[i-1]) > threshold * stdFilter[i-1]:
            if y[i] > avgFilter[i-1]:
                signals[i] = 1
            else:
                signals[i] = -1

            filteredY[i] = influence * y[i] + (1 - influence) * filteredY[i-1]
            avgFilter[i] = np.mean(filteredY[(i-lag+1):i+1])
            stdFilter[i] = np.std(filteredY[(i-lag+1):i+1])
        else:
            signals[i] = 0
            filteredY[i] = y[i]
            avgFilter[i] = np.mean(filteredY[(i-lag+1):i+1])
            stdFilter[i] = np.std(filteredY[(i-lag+1):i+1])

    return dict(signals=np.asarray(signals),
                avgFilter=np.asarray(avgFilter),
                stdFilter=np.asarray(stdFilter))
