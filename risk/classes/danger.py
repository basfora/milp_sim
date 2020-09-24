"""All functions related to danger computation"""


import os
import pickle
import time

from milp_mespp.core import plot_fun as pf
from milp_mespp.core import data_fun as df, extract_info as ext
from igraph import plot
from milp_sim.risk.scripts import r_plotfun as rpf
from milp_sim.risk.classes.hazard import MyHazard
from milp_sim.risk.scripts import base_fun as bf


class MyDanger:
    """Define danger
    for ICRA simulations
    danger is constant for planning horizon"""
    """Define danger
    for ICRA simulations
    danger is constant for planning horizon"""

    def __init__(self, g, eta_true: list or int, eta_0: list or int, plot_for_me=False):
        """
        :param g : graph
        :param xi : matching scores
        :param eta_true : ground truth danger level for each vertex
        :param eta_0 : a priori probability of danger
        :param plot_for_me
         """

        # save graph
        self.g_name = ""
        self.g = None
        # name of folder for this sim: eg smoke_G9V_grid_date#_run#
        self.folder_name = ""
        # whole path + /name_folder
        self.whole_path = ""

        self.plot = plot_for_me

        # input parameters (immutable)
        # vertices
        self.V, self.n = [], 0

        # danger threshold for the searchers
        self.kappa = list()
        self.alpha = list()

        # a priori knowledge of danger
        # eta_priori[v] = [eta_l1,... eta_l5]
        self.eta_priori = []
        # eta_priori_hat[v] = [level], level in {1,...5}
        self.eta_priori_hat = []

        # ground truth eta_check[v] \in {1,..5}
        self.eta_check = []

        # matching scores xi[v] = [[i1_1...i1_5], [i2_1,...i2_5]]
        self.xi = dict()
        # look up of estimated danger for each vertex (if robots were seeing everything)
        self.look_up_eta = []
        self.look_up_etahat = []

        # STORAGE (will be updated)
        # probability of each danger level: eta[v,t] = [prob]
        self.eta = []
        # probable danger level, highest eta: eta_hat[v,t] = max eta
        self.eta_hat = []
        # cumulative danger
        self.H = []

        # ---------------------
        # pre defined parameters
        # ---------------------
        # danger levels
        self.levels, self.n_levels, self.level_label, self.level_color = self.define_danger_levels()
        self.options = self.define_danger_perceptions()
        # ----------------------

        self.structure(g)
        self.set_danger_values(eta_true, eta_0)
        self.set_danger_thresholds()
        self.set_scores('node_score_dict_Fire.p')
        self.set_lookup_danger()
        self.perception = self.options[0]

    # called on init
    def structure(self, g):

        self.V, self.n = ext.get_set_vertices(g)

        self.g_name = g["name"]
        self.g = g

    def set_danger_values(self, eta_true, eta_0):
        self.set_priori(eta_0)
        self.set_eta_check(eta_true)

    def set_danger_thresholds(self, kappa=3, alpha=0.95):
        self.kappa = kappa
        self.alpha = alpha

    def set_priori(self, eta_0):

        n = self.n

        eta, etahat = self.danger_priori(n, eta_0)

        # eta_priori[v] = [eta_l1,... eta_l5]
        self.eta_priori = eta
        # eta_priori_hat[v] = [level], level in {1,...5}
        self.eta_priori_hat = etahat

        return

    def set_eta_check(self, eta_true):

        eta_check = self.danger_true(self.n, eta_true)

        # eta_priori[v] = [eta_l1,... eta_l5]
        self.eta_check = eta_check

    def set_scores(self, xi: str or dict):

        if isinstance(xi, str):
            my_xi = MyDanger.load_scores(xi)
        else:
            my_xi = xi

        self.xi = my_xi

    def set_lookup_danger(self, op=1):

        eta, eta_hat = [], []

        if op == 1:
            # if computing from matching scores
            xi = self.xi
            eta, eta_hat = self.compute_frequentist(xi)
        elif op == 2:
            eta_hat = self.eta_check
            eta = []
            for v in range(self.n):
                eta.append(MyDanger.eta_from_level(eta_hat[v]))
        else:
            exit()

        self.look_up_eta = eta
        self.look_up_etahat = eta_hat

        self.eta = self.look_up_eta
        self.eta_hat = self.look_up_etahat

    def set_perception(self, option: int or str):
        if isinstance(option, int):
            self.perception = self.options[option]

        elif isinstance(option, str):
            self.perception = option

    @staticmethod
    def danger_priori(n, eta_0=1):
        eta, etahat = [], []

        if isinstance(eta_0, int):
            for v in range(n):
                eta.append(MyDanger.eta_from_level(eta_0))
                etahat.append(eta_0)

        elif isinstance(eta_0, list):
            # list of lists (giving eta and not etahat)
            if isinstance(eta_0[0], list):
                eta = eta_0
                for v in range(n):
                    etahat_v = MyDanger.level_from_eta(eta[v])
                    etahat.append(etahat_v)
            else:
                etahat = eta_0
                for v in range(n):
                    eta_v = MyDanger.eta_from_level(etahat[v])
                    eta.append(eta_v)

        else:
            print('Wrong input for a priori danger values')
            exit()

        return eta, etahat

    @staticmethod
    def danger_true(n, eta_true=3):
        eta_check = []

        if isinstance(eta_true, int):
            for v in range(n):
                eta_check.append(eta_true)

        elif isinstance(eta_true, list):
            eta_check = eta_true

        else:
            print('Wrong input for ground truth danger values')
            exit()

        return eta_check

    @staticmethod
    def eta_from_level(level: int):
        """return list"""
        # TODO change for discrete distribution as level as mean
        default_prob = [0, 0, 0, 0, 0]

        default_prob[level - 1] = 1.0
        eta = default_prob

        return eta

    @staticmethod
    def level_from_eta(eta: list):
        """eta_hat = max prob level
        eta = [eta_l1,... eta_l5]

        return integer"""
        eta_hat = eta.index(max(eta)) + 1

        return eta_hat

    @staticmethod
    def compute_H(eta: list):
        """H = sum_1^l eta_l
        return list"""

        H = []
        L = len(eta)

        for i in range(L):
            H.append(sum(eta[:i]))

        return H

    @staticmethod
    def load_scores(f_name='node_score_dict_Fire.p'):
        """Load pickle file with similarity scores"""
        # load file
        folder_path = bf.get_folder_path('milp_sim', 'scores', 'risk')

        # xi_data[v] = [i1, i2, i3...], i = [xi_1, xi_2, xi_3..]
        xi_data = df.load_data(folder_path, f_name)

        return xi_data

    @staticmethod
    def compute_frequentist(xi):
        """P(L=l) = sum_{I, C} xi/sum{L,I,C}
        xi[v] = [i1, i2,...], i1 = [l1, l2, l3, l4, l5]"""

        n_l = 5

        V = [v for v in xi.keys()]
        L = list(range(1, n_l + 1))

        # pt estimate
        eta_hat = []
        # probabilities
        eta = []
        for v in V:
            # init
            sum_l = [0 for level in L]
            sum_all = 0

            # scores for this node
            for img in xi[v]:
                xi_vi = img

                for level in L:
                    idx = ext.get_python_idx(level)
                    xi_l = abs(xi_vi[idx])
                    sum_l[idx] += xi_l
                    sum_all += xi_l

            eta_hat.append(sum_l.index(max(sum_l)) + 1)
            eta.append([round(el * (1 / sum_all), 4) for el in sum_l])

        return eta, eta_hat

    @staticmethod
    def define_danger_levels():
        """Danger level notation for all milp_sim"""
        danger_levels = [1, 2, 3, 4, 5]
        n_levels = len(danger_levels)
        level_label = ['Low', 'Moderate', 'High', 'Very High', 'Extreme']
        level_color = ['green', 'blue', 'yellow', 'orange', 'red']

        return danger_levels, n_levels, level_label, level_color

    @staticmethod
    def define_danger_perceptions():
        """Point estimate or pdf"""
        perception_list = ['point', 'prob', 'none']

        return perception_list


