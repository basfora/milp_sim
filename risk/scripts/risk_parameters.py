from milp_sim.risk.classes.child_mespp import MyInputs2, MySearcher2, MySolverData2
from milp_sim.risk.classes.danger import MyDanger

from milp_mespp.core import create_parameters as cp
from milp_mespp.core import extract_info as ext


def set_default_specs():
    """Define specs for simulation of SS-2
    m = 3
    """
    specs = MyInputs2()

    # load graph, either by number (int), iGraph object or .p file name (str)
    specs.set_graph(8)
    # solver parameter: central x distributed
    specs.set_solver_type('distributed')
    # target motion
    specs.set_target_motion('static')
    # searchers' detection: capture range and false negatives
    m = 3
    specs.set_capture_range(0)
    specs.set_size_team(m)

    # time-step stuff: deadline mission (tau), planning horizon (h), re-plan frequency (theta)
    h = 5
    specs.set_all_times(h)
    specs.set_theta(1)
    # solver timeout (in sec)
    specs.set_timeout(10)

    specs.update_default()

    # danger stuff (uncomment to change)
    # specs.set_threshold(value, 'kappa')
    # specs.set_threshold(value, 'alpha')
    # specs.set_danger_data(eta_priori, eta_check)

    return specs


def create_danger(specs):
    """ create danger class"""

    g = specs.graph
    eta_true = specs.eta_check
    eta_priori = specs.eta_priori
    eta_priori_hat = specs.eta_priori_hat

    danger = MyDanger(g, eta_true, eta_priori)
    danger.set_danger_thresholds(specs.kappa, specs.alpha)
    danger.set_perception(specs.perception)

    return danger


def create_solver_data(specs):

    """Initialize solver data class from specs (exp_inputs)"""
    g = specs.graph

    # planning stuff
    deadline = specs.deadline
    theta = specs.theta
    horizon = specs.horizon
    solver_type = specs.solver_type
    timeout = specs.timeout

    solver_data = MySolverData2(horizon, deadline, theta, g, solver_type, timeout)

    return solver_data


def create_dict_searchers(g, v0: list, kappa: list, alpha: list, capture_range=0, zeta=None):
    """Create searchers (dictionary with id number as keys).
            Nested: initial position, capture matrices for each vertex"""

    # set of searchers S = {1,..m}
    S = ext.get_set_searchers(v0)[0]
    # create dict
    searchers = {}
    for s_id in S:
        v = ext.get_v0_s(v0, s_id)
        cap_s = ext.get_capture_range_s(capture_range, s_id)
        zeta_s = ext.get_zeta_s(zeta, s_id)

        # create each searcher
        s = MySearcher2(s_id, v, g, cap_s, zeta_s)
        s.set_alpha(alpha[s_id - 1])
        s.set_kappa(kappa[s_id - 1])

        # store in dictionary
        searchers[s_id] = s

    return searchers


def create_searchers(specs):

    # unpack from specs
    g = specs.graph
    capture_range = specs.capture_range
    zeta = specs.zeta
    m = specs.size_team
    alpha = specs.alpha
    kappa = specs.kappa

    if specs.start_searcher_v is None:
        # if initial position was not defined by user
        v_list = cp.placement_list(specs, 's')
        if specs.searcher_together:
            v_list = cp.searchers_start_together(m, v_list)
        # len(v0) = m
        v0 = v_list
        specs.set_start_searchers(v0)
    else:
        # if it was, use that
        v0 = specs.start_searcher_v

    # get graph vertices
    V, n = ext.get_set_vertices(g)
    if any(v0) not in V:
        print("Vertex out of range, V = {1, 2...n}")
        return None

    searchers = create_dict_searchers(g, v0, kappa, alpha, capture_range, zeta)

    return searchers


def get_kappa(searchers: dict):
    """Retrieve kappa from each searcher and return as list"""

    kappa_list = []

    for s_id in searchers.keys():
        s = searchers[s_id]
        kappa_list.append(s.kappa)

    return kappa_list


def get_alpha(searchers:dict):
    """Retrieve alpha from each searcher and return as list"""

    alpha_list = []

    for s_id in searchers.keys():
        s = searchers[s_id]
        alpha_list.append(s.alpha)

    return alpha_list


def get_H(danger, searchers):
    """list_H : list of cumulative danger probability,
    list_H = [H_v1, H_v2,...,H_vn], H_v1 = [H_s1, H_s2.. H_sm], H_s1 = sum_{l=1}^k eta_l, H_s1 in [0,1]"""

    list_H = []

    for v in range(danger.n):
        eta_v = danger.eta[v]
        H_v = []
        H_l = danger.compute_H(eta_v)
        for s_id in searchers.keys():
            kappa_idx = searchers[s_id].kappa - 1
            H_s = sum(H_l[:kappa_idx])
            H_v.append(H_s)
        list_H.append(H_v)

    return list_H











