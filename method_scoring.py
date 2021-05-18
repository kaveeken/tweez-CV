import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy

from running_fit import running_fit, build_handles_model
from event_finding import find_transitions, spline_residuals
from beter_sim import full_sim, handle_estimates, protein_estimates

def parse_results(true_unfolds, unfolds, width=2):
    matches = []
    orig_unfolds = deepcopy(unfolds)
    for true_unfold in true_unfolds:
        for unfold in unfolds:
            if unfold - width <= true_unfold <= unfold + width:
                matches.append(unfold)
                unfolds.remove(unfold)
                break
    offs = [min([abs(true_unfold - unfold) for true_unfold in true_unfolds])
                for unfold in unfolds]

    return {'matches': len(matches),
            'false_pos': len(unfolds),
            'misses': len(true_unfolds) -len(matches),
            'off_by': offs,
            'true': true_unfolds,
            'found': orig_unfolds,
            'matched': matches,
            'fp': unfolds}


def test_runfit(sim, first_index=0, last_index=-1,
                slice_width=50, model=build_handles_model(),
                lag_factor=3, threshold=3., influence=0):
    true_unfolds = sim[1]
    unfolds = running_fit(sim[0][1], sim[0][0],
                          first_index, last_index,
                          slice_width, model,
                          lag_factor, threshold, influence)['unfolds']

    plt.figure()
    plt.plot(sim[0][1])
    for unfold in unfolds:
        plt.axvline(unfold)
    print(true_unfolds)
    print(unfolds)
    return parse_results(true_unfolds, unfolds)

def test_find_transitions(sim):
    f = sim[0][1]
    true_unfolds = sim[1]
    unfolds, thresholds = find_transitions(f)
    return parse_results(true_unfolds, list(unfolds))

def test_spline_residuals(sim):
    r = spline_residuals(sim[0][1])
    return test_find_transitions(((sim[0][0], r), sim[1]))


def bulk_test(force_sigmas, dist_sigmas,
              cls=[0.005 * i for i in range(1,7)],
              ufs=[30 + 5 * i for i in range(6)],
              replicates = 10, skip_runfit=False):
    results = []
    sims = [full_sim(ufs, cls, handle_estimates, protein_estimates)
            for i in range(replicates)]
    for force_sigma, dist_sigma in zip(force_sigmas, dist_sigmas):
        local_results = {
            'force_sigma': force_sigma,
            'dist_sigma': dist_sigma,
            'vanilla': {'matches': 0,
                        'misses': 0,
                        'false_pos': 0},
            'spline': {'matches': 0,
                       'misses': 0,
                       'false_pos': 0},
            'runfit': {'matches': 0,
                       'misses': 0,
                       'false_pos': 0}}
        if skip_runfit:
            local_results.pop('runfit')
        for sim in sims:
            noisy_d = sim[0][0] + np.random.normal(0, dist_sigma, sim[0][0].shape)
            noisy_f = sim[0][1] + np.random.normal(0, force_sigma, sim[0][1].shape)
            noised_sim = ((noisy_d, noisy_f), sim[1])
            vanilla = test_find_transitions(noised_sim)
            spline = test_spline_residuals(noised_sim)
            if not skip_runfit:
                runfit = test_runfit(noised_sim)
                for key in local_results['runfit'].keys():
                    local_results['runfit'][key] += spline[key]
            for key in local_results['vanilla'].keys():
                local_results['vanilla'][key] += vanilla[key]
            for key in local_results['spline'].keys():
                local_results['spline'][key] += spline[key]

        results.append(local_results)
    return results


def write_results(fname, results):
    with open(fname, 'w') as f:
        f.write('sigma force, method, matches, false negative, false positive, score\n')
        for result in results:
            vanilla_score = (result['vanilla']['matches']
                             - result['vanilla']['false_pos']) \
                             / (result['vanilla']['matches']
                                + result['vanilla']['misses'])
            spline_score = (result['spline']['matches']
                             - result['spline']['false_pos']) \
                             / (result['spline']['matches']
                                + result['spline']['misses'])
            runfit_score = (result['runfit']['matches']
                             - result['runfit']['false_pos']) \
                             / (result['runfit']['matches']
                                + result['runfit']['misses'])
            f.write(f"{result['force_sigma']}, vanilla, {result['vanilla']['matches']}, {result['vanilla']['misses']}, {result['vanilla']['false_pos']},{round(vanilla_score,2)}\n")
            f.write(f", spline, {result['spline']['matches']}, {result['spline']['misses']}, {result['spline']['false_pos']},{round(spline_score,2)}\n")
            f.write(f", runfit, {result['runfit']['matches']}, {result['runfit']['misses']}, {result['runfit']['false_pos']},{round(runfit_score,2)}\n")


def write_summary(fname, results):
    with open(fname, 'w') as f:
        f.write('STD, NOT, SRNOT, RF\n')
        for result in results:
            scores = {}
            for key in ['vanilla', 'spline', 'runfit']:
                if key not in result:
                    break
                scores[key] = round((result[key]['matches']
                               - result[key]['false_pos']) \
                               / (result[key]['matches']
                                  + result[key]['misses']), 3)
            towrite = str(result['force_sigma'])
            for key in scores:
                towrite += ',' + str(scores[key])
                
            f.write(towrite + '\n')
    
