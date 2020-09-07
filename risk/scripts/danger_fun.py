"""Danger-related functions
 - add constraints (prob or point estimate)
 - add constraints
 - set objective function
 - solve model
 - retrieve model data
 - query variables

OBS: s in {1,...m}
G(V, E), v in {1,...n} but igraph treats as zero
t in tau = {1,...T}
capture matrix: first index is zero [0][0], first vertex is [1][1]
v0 can be {1,...n} --> I convert to python indexing later
array order: [s, t, v, u]"""

from milp_mespp.core import extract_info as ext
from milp_mespp.core import milp_fun as mf
from milp_mespp.core import data_fun as df
from gurobipy import *
import os
import pickle


# -----------------------------------------------------------------------------------
# Solver functions
# -----------------------------------------------------------------------------------
def add_kappa_point(md, my_vars: dict, list_hat_eta: list, list_kappa: list, horizon: int):
    """
    :param md : Gurobi model
    :param horizon : planning horizon (deadline)
    :param my_vars : model variables

    :param list_hat_eta : list of estimated danger,
    list_hat_eta = [eta_v1, eta_v2, ..., eta_vn], hat_eta in L = {1, ...5}, hat_eta = max(eta_v,l^t)

    :param list_kappa : list of danger threshold for each searcher,
    list_kappa = [k_s1, k_s2...k_sm], kappa in L = {1, 2, 3, 4, 5}

    OBS: eta is constant for planning horizon
    """

    # get variables
    X = mf.get_var(my_vars, 'x')

    n = len(list_hat_eta)
    m = len(list_kappa)

    V = ext.get_set_vertices(n)[0]
    S = ext.get_set_searchers(m)[0]
    T = ext.get_set_time(horizon)

    for v in V:
        v_idx = ext.get_python_idx(v)
        # estimate danger level (eta_hat in L = {1,...,5})
        eta_hat = list_hat_eta[v_idx]

        for s in S:
            s_idx = ext.get_python_idx(s)
            # danger threshold (kappa in L = {1,...5})
            kappa = list_kappa[s_idx]

            for t in T:
                md.addConstr(X[s, v, t] * eta_hat <= kappa)


def add_kappa_prob(md, my_vars: dict, list_H: list, list_alpha: list, horizon: int):
    """
    :param md : Gurobi model
    :param horizon : planning horizon (deadline)
    :param my_vars : model variables

    :param list_H : list of cumulative danger probability,
    list_H = [H_v1, H_v2,...,H_vn], H_v1 = [H_s1, H_s2.. H_sm], H_s1 = sum_{l=1}^k eta_l, H_s1 in [0,1]

    :param list_alpha : list of danger confidence threshold for each searcher,
    list_alpha = [a_s1, a_s2...a_sm], alpha in [0,1]

    OBS: eta is constant for planning horizon
    """

    # get variables
    X = mf.get_var(my_vars, 'x')

    n = len(list_H)
    m = len(list_alpha)

    V = ext.get_set_vertices(n)[0]
    S = ext.get_set_searchers(m)[0]
    T = ext.get_set_time(horizon)

    for v in V:
        v_idx = ext.get_python_idx(v)
        # cumulative danger level - list
        H_v = list_H[v_idx]

        for s in S:
            s_idx = ext.get_python_idx(s)
            # danger threshold - cumulative - for searcher s
            H_s = H_v[s_idx]
            # danger threshold - confidence - for searcher s
            alpha = list_alpha[s_idx]

            for t in T:
                md.addConstr(H_s >= X[s, v, t] * alpha)


# -----------------------------------------------------------------------------------
# Extracting danger functions
# -----------------------------------------------------------------------------------
def load_scores():

    # load file
    f_name = 'sim_score.pkl'
    folder_path = ext.get_folder_path('milp_sim', 'scores', 'risk')

    # xi_data[v] = [i1, i2, i3...], i = [xi_1, xi_2, xi_3..]
    xi_data = df.load_data(folder_path, f_name)

    return xi_data

# TODO xi --> eta (points system)


# -----------------------------------------------------------------------------------
# Plan functions
# -----------------------------------------------------------------------------------

# TODO same as milp_mespp.plan but with added inputs and calling add_constraints