import numpy as np
import lumicks.pylake as lk
from matplotlib import pyplot as plt
from copy import deepcopy

from util import load_estimates

def gen_hm():
    return lk.inverted_odijk('handles') + lk.force_offset('handles')
def gen_comp():
    comp_wrt_f = lk.odijk('handles') + lk.inverted_marko_siggia_simplified('protein')
    return comp_wrt_f.invert(interpolate = True, 
                             independent_min = 0,
                            independent_max = 90) + lk.force_offset('handles')

def generate_fd(first_unf, cls, handle_estimates_orig, protein_estimates_orig,
                stds={'dist': 0.00195, 'force': 0.105}):
    model_h = gen_hm()
    model_c = gen_comp()
    fit = lk.FdFit(model_h, model_c)
    junk = (np.linspace(0.28,0.3,100), np.linspace(0,20,100))
    fit[model_h].add_data('junk', *junk)
    fit[model_c].add_data('junk', *junk)

    handle_estimates = deepcopy(handle_estimates_orig)
    protein_estimates = deepcopy(protein_estimates_orig)
    for param in handle_estimates.values():
        param['fixed'] = True
    for param in protein_estimates.values():
        param['fixed'] = True
    load_estimates(fit, handle_estimates)
    load_estimates(fit, protein_estimates)
    fit['protein/Lc'].value = cls[0]

    unfold_dist = first_unf
    stop_dist = unfold_dist + sum(cls)
    pull_dists = np.linspace(0.28, stop_dist, 2000)
    distances = [pull_dists[pull_dists < unfold_dist]]
    forces = [model_h(distances[0], fit)]
    total_cl = 0
    for index, cl in enumerate(cls):
        newdist = pull_dists[pull_dists >= unfold_dist]
        # print(len(newdist))
        unfold_dist += cl
        # print(index, cl, len(cls))
        if index < len(cls) - 1:
            newdist = newdist[newdist < unfold_dist]
        distances.append(newdist)
        total_cl += cl
        fit['protein/Lc'].value = total_cl
        newforce = model_c(newdist, fit)
        forces.append(newforce)
    stat_dists = np.zeros(500) + stop_dist
    stat_forces = np.zeros(500) + model_c(stop_dist, fit)
    distances.append(stat_dists)
    forces.append(stat_forces)
    relax_dists = np.linspace(stop_dist, 0.28, 2000) 
    relax_forces = model_c(relax_dists, fit)
    distances.append(relax_dists)
    forces.append(relax_forces)
    concdists = np.concatenate(distances, axis=None)
    concforces = np.concatenate(forces, axis=None)
    noisy_dists = concdists + np.random.normal(0, stds['dist'], len(concdists))
    noisy_forces = concforces + np.random.normal(0, stds['force'], len(concforces))
    return (noisy_dists, noisy_forces)


#gens = generate_fd(0.38, [0.020, 0.025, 0.035])

# plt.figure(figsize=(8,14))
# plt.subplot(2, 1, 1)
# plt.plot(*gens, '.', c='tab:blue')
# plt.subplot(2, 1, 2)
# plt.plot(gens[1], '.', c='tab:blue')
# plt.savefig('sim_test.png')
