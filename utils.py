import argparse
import numpy as np
import scipy as sp
import scipy.stats
import matplotlib.pyplot as plt

class ArgsParser:
    """
    Read the user's input and parse the arguments properly. When returning args, each value is properly filled.
    Ideally one shouldn't have to read this function to access the proper arguments, but I postpone this.
    """
    @staticmethod
    def read_input_args():
        # Parse command line
        parser = argparse.ArgumentParser(
            description='Define algorithm\'s parameters.')

        parser.add_argument('-e', '--env_name', type=str, default='room',
                            help='Environment name. See environment.py')

        parser.add_argument('-o', '--output', type=str, default='output/',
                            help='Prefix that will be used to generate all outputs (default: output/).')

        parser.add_argument('-t', '--n_trials', type=int, default=1,
                            help='Number of trials to be averaged over when appropriate (default: 1).')

        parser.add_argument('-s', '--n_steps', type=int, default=50,
                            help='Maximum number of time steps an episode may last (default: 50).')

        parser.add_argument('-n', '--n_episodes', type=int, default=1000,
                            help='Number of episodes in which learning will happen (default: 1000).')

        args = parser.parse_args()

        return args

def mean_confidence_interval(data, confidence=0.95):
    """
    Code obtained from the link below:
    https://stackoverflow.com/questions/15033511/compute-a-confidence-interval-from-sample-data
    """
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a, axis=0), scipy.stats.sem(a, axis=0)
    h = se * sp.stats.t._ppf((1+confidence)/2., n-1)
    return m, m-h, m+h
