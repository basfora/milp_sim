from milp_sim.risk.classes.child_mespp import MyInputs2, MySolverData2, MyMission
from milp_sim.risk.classes.team import MyTeam2
from milp_sim.risk.classes.danger import MyDanger

from milp_mespp.core import create_parameters as cp
from milp_mespp.core import extract_info as ext


def default_specs():
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

    specs.use_default()

    # danger stuff (uncomment to change)
    # specs.set_threshold(value, 'kappa')
    # specs.set_threshold(value, 'alpha')
    # specs.set_danger_data(eta_priori, eta_check)

    return specs


def create_danger(specs):
    """ create danger class"""

    g = specs.graph
    # create danger class
    danger = MyDanger(g)

    # perception: point or prob
    danger.set_perception(specs.perception)

    # thresholds
    danger.set_thresholds(specs.kappa, specs.alpha)

    # kill prob
    danger.set_kill(specs.danger_kill, specs.prob_kill)

    # constraints (use or not)
    danger.set_constraints(specs.danger_constraints)

    # how to compute z from eta
    danger.set_mva_conservative(specs.mva_conservative)

    # if you are using fov
    danger.set_use_fov(specs.fov)

    # true priori knowledge
    danger.set_true_priori(specs.true_priori)

    # parse actual values or files
    danger.set_true(specs.danger_true)

    if specs.true_priori is False:
        # if danger hat is None, it's gonna use the true values
        danger.set_estimate(specs.danger_hat)
        danger.set_priori(specs.danger_priori)
    else:
        danger.uniform_priori = False

    return danger


def create_mission(specs):
    mission = MyMission()

    # team info
    mission.set_team_size(specs.size_team)
    mission.set_team_thresholds(specs.kappa)

    # planning info
    mission.set_deadline(specs.deadline, specs.horizon)

    # danger info
    mission.define_danger(specs.danger_constraints, specs.danger_kill, specs.prob_kill)
    mission.set_perception(specs.perception, specs.percentage_img)

    return mission


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


# -----------------------------------------
# searchers-related functions
# -----------------------------------------
def create_searchers(specs):
    # TODO move this to team class later

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

    team = MyTeam2()
    team.create_dict_searchers(g, v0, kappa, alpha, capture_range, zeta)
    team.set_danger_perception(specs.perception)

    return team


# deprecated - into team
def get_kappa(searchers: dict):
    """Retrieve kappa from each searcher and return as list"""

    kappa_list = []

    for s_id in searchers.keys():
        s = searchers[s_id]
        kappa_list.append(s.kappa)

    return kappa_list


# deprecated - into team
def get_alpha(searchers: dict):
    """Retrieve alpha from each searcher and return as list"""

    alpha_list = []

    for s_id in searchers.keys():
        s = searchers[s_id]
        alpha_list.append(s.alpha)

    return alpha_list


# deprecated - into team
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



















