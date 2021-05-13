import numpy as np
import lumicks.pylake as lk

from util import load_estimates

_hm_f = lk.odijk('handles')
_comp_f = lk.odijk('handles') \
    + lk.inverted_marko_siggia_simplified('protein')

_hm = lk.inverted_odijk('handles')
_comp = _comp_f.invert(interpolate=True,
                       independent_min=0,
                       independent_max=90)

handle_estimates = \
    {'handles/Lp':  # DNA handle persistence length (nm)
      {'value': 15,  # initial estimate
       'upper_bound': 100,  # very wide bounds?
       'lower_bound': 0.0},
     'handles/Lc':  # contour length (um)
      {'value': 0.3},#bp2cl(1040)},  # bp2cl generates a contour length from a number of basepairs.
     'handles/St':  # stretch modulus (pN)
      {'value': 300,
       'lower_bound': 250}
  }
protein_estimates = \
     {'protein/Lp':  # unfolded protein persistence length (nm)
      {'value': 0.7,
      'upper_bound': 1.0,
      'lower_bound': 0.6,
      'fixed': False},
     'protein/Lc':  # contour length (um)
      {'value': 0.001,
       'fixed': False}
     }

def compute_unfold_distances(ufs, cls, handle_estimates,
                             protein_estimates):
    fits = [lk.FdFit(_hm_f),
            *[lk.FdFit(_hm_f, _comp_f) for uf in ufs]]
    fits[0][_hm_f].add_data('junk', [0], [0])
    load_estimates(fits[0], handle_estimates)
    low_force = _hm_f(5, fits[0])
    uds = [_hm_f(ufs[0], fits[0])]

    cl_total = 0
    for fit, cl, uf in zip(fits[1:], cls[:-1], ufs[1:]):
        fit[_comp_f].add_data('junk', [0], [0])
        cl_total += cl
        load_estimates(fit, handle_estimates)
        load_estimates(fit, protein_estimates)
        fit['protein/Lc'].value = cl_total
        uds.append(_comp_f(uf, fit))
    return (low_force, uds)


def simulate_distance(low, uds, cls, length=2000):
    top = uds[-1] + max(np.diff(uds))
    graph = [low, *uds, top]
    for index, cl in enumerate(cls):
        location = (index + 1) * 2
        value = graph[location - 1] + cl / 2
        graph.insert(location, value)
    steps = np.diff(graph)
    total = sum(steps)
    legs = []
    for index, node in enumerate(graph):
        if index % 2:
            weight = (node - graph[index - 1]) / total
            leg_len = int(weight * length)
            legs.append(np.linspace(graph[index - 1], node, leg_len))
    return legs


def simulate_force(d_legs, cls, handle_estimates, protein_estimates):
    fit = lk.FdFit(_hm)
    fit[_hm].add_data('junk', [0], [0])
    load_estimates(fit, handle_estimates)
    f_legs = [_hm(d_legs[0], fit)]
    cl_total = 0
    for d_leg, cl in zip(d_legs[1:], cls):
        fit = lk.FdFit(_hm, _comp)
        fit[_comp].add_data('junk', [0], [0])
        load_estimates(fit, handle_estimates)
        load_estimates(fit, protein_estimates)
        cl_total += cl
        fit['protein/Lc'].value = cl_total
        f_legs.append(_comp(d_leg, fit))
    unfold_points = [len(d_legs[0])]
    for leg in d_legs[1:]:
        unfold_points.append(unfold_points[-1] + len(leg))
    return ((np.concatenate(d_legs), np.concatenate(f_legs)), unfold_points)


def full_sim(ufs, cls, handle_estimates, protein_estimates):
    (low, uds) = compute_unfold_distances(ufs, cls, handle_estimates,
                                          protein_estimates)
    d_legs = simulate_distance(low, uds, cls)
    return simulate_force(d_legs, cls, handle_estimates, protein_estimates)


if __name__ == '__main__':
    from matplotlib import pyplot as plt
    from ests import handle_estimates, protein_estimates
    ufs = [30, 40, 50]
    cls = [0.01, 0.015, 0.01]
    dists = compute_unfold_distances(ufs, cls,
                                     handle_estimates, protein_estimates)
    print(dists)
    d_legs = simulate_distance(*dists, cls)
    (d, f) = simulate_force(d_legs, cls, handle_estimates, protein_estimates)
    fig, ax1 = plt.subplots()
    ax1.plot(f)
    for uf in ufs:
        ax1.axhline(uf)
    ax2 = ax1.twinx()
    ax2.plot(d, c='tab:orange')
    plt.show()
