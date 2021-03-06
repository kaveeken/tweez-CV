import lumicks.pylake as lk
import numpy as np
from math import exp
from matplotlib import pyplot as plt

from event_finding import get_first_trough_index, find_transitions, top_finder
from event_finding import spline_residuals
from util import extract_estimates, load_estimates


class Curve:
    """ This class holds the data of an FD curve and methods used for analysis.
    Many methods rely on internal state change and can only be ran sequentially.
    
    Now takes separate pull and relax data because it's easier to handle.
    Rest of class still needs some adapting.
    """
    def __init__(self, identifier: str, ddata: np.ndarray, fdata: np.ndarray,
                 pull_d=np.empty([]), pull_f=np.empty([]),
                 rlx_d=np.empty([]), rlx_f=np.empty([])):
        """ Take the identifier, distance data and force data for an fd curve.
        """
        self.identifier = identifier
        self.dist_data = ddata
        self.force_data = fdata

        self.split = 0
        if pull_f.size > 2 and pull_d.size > 2:
            self.split += 1
            self.pull_f = pull_f
            self.pull_d = pull_d
            if rlx_f.any() and rlx_d.any():
                self.split += 1
                self.rlx_f = rlx_f
                self.rlx_d = rlx_d

    def filter_bead_loss(self, handle_contour=False):
        """ Test for a sudden drop in force to 0. Implementation is somewhat
        arbitrary. Returns True if a sudden drop is detected and False
        otherwise.
        
        The `handle_contour` argument governs whether to apply a method 
        that uses a DNA handle contour estimate.
        """
        if handle_contour:
            start = np.argwhere(self.dist_data > handle_contour)[0][0]
            floor = np.min(self.force_data[start:start+100])
        else:
            floor = 10 # arbitrary and feels bad same as 40 below
        for index, force in enumerate(self.force_data[1:]):
            if force <= floor and self.force_data[index] > 40:
                return True
        return False

    def find_events(self, STARTING_FORCE: float = 0, CORDON: int = 10,
                    FIT_ON_RETURN: tuple = (), DEBUG: bool = False,
                    LONGEST=1e6, handle_contour=False, splines=True):
        """ Identifies relevant events in the force data over time. From those
        events determines which parts (legs) of the data to mark for fitting.
        These legs (slice objects) are put in a list `self.legs`.

        # Arguments:
        - STARTING_FORCE: we do not mark for fitting data before the point
        where STARTING_FORCE is reached.
        - CORDON: we exclude from fitting datapoints within CORDON points
        before and after an event.
        - FIT_ON_RETURN: tuple describing an arbitrary part of the
        relaxation curve to mark for fitting. Should either be empty or contain
        two numbers describing how far after the return point and how many
        datapoints to include.
        - DEBUG: if something goes wrong with identifying the return point or
        stationary part of the curve, set this to True to print specific info.
        """
        if DEBUG:
            print(self.identifier)
            print(self.split)
        if not self.split:  # old part
            self.top = (get_first_trough_index(self.force_data, debug=DEBUG),
                        get_first_trough_index(self.force_data, last=True,
                                               debug=DEBUG))
            self.pull_d = self.dist_data[:self.top[0]]
            self.pull_f = self.dforce_data[:self.top[0]]
            if self.top[1] - self.top[0] > 100 and False:
                self.unfolds, self.threshold = \
                    find_transitions(self.force_data,
                                     noise_estimation_window=self.top)
            else:
                self.unfolds, self.threshold = find_transitions(self.pull_f)

                self.start = 0
                for index, force in enumerate(self.force_data):
                    if force > STARTING_FORCE:
                        self.start = index
                        break
                    halfway = int(len(self.force_data) / 2)
                    force_min = np.argmin(self.force_data[:halfway])
                    self.start = force_min

                    events = [self.start, *self.unfolds, self.top[0]]
                    self.legs = [slice(*[events[i] + CORDON,
                                         min(events[i+1] - CORDON,
                                             events[i] + CORDON + LONGEST)])
                                 for i in range(len(events) - 1)]

                    if FIT_ON_RETURN:
                        self.legs.append(slice(self.top_window[-1] + FIT_ON_RETURN[0],
                                               self.top_window[-1] + sum(FIT_ON_RETURN)))
        else:
            self.start = 0
            if handle_contour:
                self.start = np.argwhere(self.pull_d > handle_contour)[0][0]
            else:
                self.start = np.argwhere(self.pull_f > STARTING_FORCE)[0][0]
            # pull_f should be turned into pull_f[start:]?
            residuals = spline_residuals(self.pull_f[self.start:])
            if splines:
                self.unfolds, self.threshold = \
                    find_transitions(residuals)
            else:
                self.unfolds, self.threshold = \
                    find_transitions(self.pull_f[self.start:])
            self.unfolds = np.asarray([unfold + self.start for unfold in self.unfolds])
            events = [self.start, *self.unfolds, len(self.pull_f)]
            self.legs = [slice(*[events[i] + CORDON,
                                 min(events[i+1] - CORDON,
                                     events[i] + CORDON + LONGEST)])
                         for i in range(len(events) - 1)]



    def plot_events(self):
        fig = plt.figure()
        N = len(self.force_data)
        plt.plot(np.arange(N), self.force_data, c='tab:blue')
        for unfold in self.unfolds:
            plt.plot(np.arange(unfold, unfold + 5),
                     self.force_data[unfold: unfold+5], c='tab:orange')
        for leg in self.legs:
            plt.plot(np.arange(N)[leg],  # np.arange(leg) ?
                     self.force_data[leg],
                     c='tab:green')
        if not self.split:
            plt.plot(np.arange(self.top[0], self.top[1]),
                    self.force_data[self.top[0]:self.top[1]], c='tab:red')
        plt.title(self.identifier)
        return fig

    def filter_tethers(self, model: lk.fitting.model.Model,
                       estimates_dict: dict, VERBOSE: bool = True):
        """ Tests for multiple tethers by comparing the fit of the raw estimates
        to changes made in those estimates that are meant to better describe
        the case of two tethers.
        Similarly tries to redescribe the system to better fit the two-tether
        case by halving force data and doubling distance data.
        Results are stored in the `self.tether_tests` dictionary with keys
        corresponding to the different tests and values `True` for failed tests
        or `False` for passed tests.

        # Arguments:
        - model: Pylake model object of the relevant (DNA handles) model.
        - estimates_dict: dictionary with the raw estimates as well as changed
        estimates for each test. Raw estimates have to be named 'original'.
        - VERBOSE: toggle to print out the test results immediately
        """
        fits = {test_id: lk.FdFit(model) for test_id in estimates_dict.keys()}
        handle_forces = self.force_data[self.legs[0]]
        handle_dists = self.dist_data[self.legs[0]]

        self.bics = {}
        for key, fit in fits.items():
            if key == 'half_force':
                fit.add_data(f'{self.identifier}_{key}',
                             handle_forces / 2, handle_dists)
                load_estimates(fit, estimates_dict['original'])
            elif key == 'double_dist':
                fit.add_data(f'{self.identifier}_{key}',
                             handle_forces, handle_dists * 2)
                load_estimates(fit, estimates_dict['original'])
            else:
                fit.add_data(f'{self.identifier}_{key}',
                             handle_forces, handle_dists)
                load_estimates(fit, estimates_dict[key])
            self.bics[key] = fit.bic

        # this computation breaks sometimes?
        # self.bfactors = \
        #     {test_id: exp((self.bics['original'] - self.bics[test_id]) / 2)
        #      for test_id in self.bics.keys() - 'original'}
        self.tether_tests = \
            {test_id: self.bics['original'] > self.bics[test_id]
             for test_id in self.bics.keys() - 'original'}

        if VERBOSE:
            print(self.identifier, '\n', self.tether_tests)

    def initialize_fits(self, handles_model: lk.fitting.model.Model,
                        composite_model: lk.fitting.model.Model,
                        handle_estimates: dict):
        """ initialize a lk.FdFit object for each unfolding event and perform
        a fit for the DNA handles part of the system. The results of that fit
        are copied onto all fit objects and fixed. A list of fits is stored
        in `self.fits`.

        # Arguments
        - handles_model: pylake model object describing the DNA handles.
        - composite_model: pylake model object describing handles + protein.
        - handle_estimates: initial estimates given for the DNA handles model.
        """
        self.composite_model = composite_model
        self.handles_model = handles_model
        self.fits = [lk.FdFit(handles_model, composite_model)\
                     for unfold in self.unfolds]
        for fit in self.fits:
            fit[handles_model].add_data(f'{self.identifier}_handles_model',
                                        self.force_data[self.legs[0]],
                                        self.dist_data[self.legs[0]])
        load_estimates(self.fits[0], handle_estimates)
        self.fits[0].fit()
        self.fits[0]['handles/St'].fixed = True
        self.fits[0]['handles/Lp'].fixed = True
        self.fits[0]['handles/Lc'].fixed = True
        self.fits[0]['handles/f_offset'].fixed = True

        for fit in self.fits[1:]:
            load_estimates(fit, extract_estimates(self.fits[0]))

    def fit_composites(self, protein_estimates: dict):
        """ Perform a fit for each remaining (post-unfold) leg using the
        composite model, fitting only the protein parameters.

        # Arguments:
        - protein_estimates: initial estimates given for the composite model.
        """
        # this first part seems sketchy
        if len(self.legs[1:]) > len(self.fits):
            legs = self.legs[1:-1]
            relax = self.legs[-1]
        else:
            legs = self.legs[1:]
            relax = False

        for index, (fit, leg) in enumerate(zip(self.fits, legs)):
            fit[self.composite_model].add_data(f'{self.identifier}_dom_{index}',
                                               self.force_data[leg],
                                               self.dist_data[leg])
            if index >= len(legs) - 1 and relax:
                self.fits[-1][self.composite_model].add_data(
                    f'{self.identifier}_relax',
                    self.force_data[relax], self.dist_data[relax])

            load_estimates(fit, protein_estimates)
            if index:  # help fit along using previous result
                prev_cl = self.fits[index - 1].params['protein/Lc'].value
                fit.params['protein/Lc'].value = prev_cl + 0.01
                fit.params['protein/Lc'].lower_bound = prev_cl

            fit.fit()

    def print_fit_params(self):
        for fit in self.fits:
            print(fit.params)

    def plot_fits(self):
        fig = plt.figure()
        plt.title(self.identifier)
        self.fits[0][self.handles_model].plot()
        for fit in self.fits:
            fit[self.composite_model].plot()
        return fig

    def compute_unfold_forces(self, handles_model: lk.fitting.model.Model,
                              composite_model: lk.fitting.model.Model,
                              VERBOSE=True):
        """ Computes unfolding forces by simulating the preceding model directly
        before the unfolding event.

        # Arguments:
        - handles_model: pylake model object for DNA handles
        - composite_model: pylake model object for handles + protein
        - VERBOSE: option to immediately print out a list of unfolding forces
        """
        unfold_dists = [self.dist_data[unfold - 1] for unfold in self.unfolds]
        self.unfold_forces = [handles_model(unfold_dists[0], self.fits[0])]
        for dist, fit in zip(unfold_dists[1:], self.fits[:-1]):
            self.unfold_forces.append(composite_model(dist, fit))
        if VERBOSE:
            print(self.identifier, '\n', self.unfold_forces)

    def print_result_rows(self, row_format: str):
        """ Print a row containing contour length, persistence length and
        unfolding force for each unfolded domain. Formatted according to
        the `row_format` argument.

        # Arguments:
        - row_format: python formatting string
        """
        total_cl = 0
        for index, fit in enumerate(self.fits):
            Lc = round(fit['protein/Lc'].value - total_cl, 6)
            Lp = round(fit['protein/Lp'].value, 6)
            Fu = round(self.unfold_forces[index], 4)
            tests = self.tether_tests
            failed_tests = \
                [test for test in tests if tests[test]]
            total_cl += Lc
            print(row_format.format(self.identifier, index + 1, Lc, Lp, Fu,
                                    failed_tests))
