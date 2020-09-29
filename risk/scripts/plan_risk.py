"""Danger-related functions
 - add constraints (prob or point estimate)
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
from milp_mespp.core import create_parameters as cp
from milp_mespp.core import construct_model as cm
from milp_mespp.core import plan_fun as pln
from gurobipy import *

from milp_sim.risk.scripts import risk_parameters as rp


# -----------------------------------------------------------------------------------
# Solver -- functions
# -----------------------------------------------------------------------------------
# UT - ok
def add_kappa_point(md, my_vars: dict, vertices_t: dict, list_hat_eta: list, list_kappa: list, horizon: int):
    """
    :param md : Gurobi model
    :param horizon : planning horizon (deadline)
    :param my_vars : model variables
    :param vertices_t : allowed vertices

    :param list_hat_eta : list of estimated danger,
    list_hat_eta = [etahat_v1, etahat_v2, ..., etahat_vn], hat_eta in L = {1, ...5}, hat_eta = max(eta_v,l^t)

    :param list_kappa : list of danger threshold for each searcher,
    list_kappa = [k_s1, k_s2...k_sm], kappa in L = {1, 2, 3, 4, 5}

    OBS: eta is constant for planning horizon
    """

    # get variables
    X = mf.get_var(my_vars, 'x')

    m = len(list_kappa)

    S = ext.get_set_searchers(m)[0]
    T = ext.get_set_time(horizon)

    for s in S:
        s_idx = ext.get_python_idx(s)
        # danger threshold (kappa in L = {1,...5})
        kappa = list_kappa[s_idx]

        for t in T:
            v_t = vertices_t.get((s, t))

            for v in v_t:
                v_idx = ext.get_python_idx(v)
                # estimate danger level (eta_hat in L = {1,...,5})
                eta_hat = list_hat_eta[v_idx]
                md.addConstr(X[s, v, t] * eta_hat <= kappa)


def add_kappa_prob(md, my_vars: dict, vertices_t: dict, list_H: list, list_alpha: list, horizon: int):
    """
    :param md : Gurobi model
    :param horizon : planning horizon (deadline)
    :param my_vars : model variables
    :param vertices_t : allowed vertices

    :param list_H : list of cumulative danger probability,
    list_H = [H_v1, H_v2,...,H_vn], H_v1 = [H_s1, H_s2.. H_sm], H_s1 = sum_{l=1}^k eta_l, H_s1 in [0,1]

    :param list_alpha : list of danger confidence threshold for each searcher,
    list_alpha = [a_s1, a_s2...a_sm], alpha in [0,1]

    OBS: eta is constant for planning horizon
    """

    # get variables
    X = mf.get_var(my_vars, 'x')

    m = len(list_alpha)

    S = ext.get_set_searchers(m)[0]
    T = ext.get_set_time(horizon)

    for s in S:
        s_idx = ext.get_python_idx(s)
        # danger threshold - confidence - for searcher s
        alpha = list_alpha[s_idx]

        for t in T:
            v_t = vertices_t.get((s, t))

            for v in v_t:
                v_idx = ext.get_python_idx(v)
                # cumulative danger level - list
                H_v = list_H[v_idx]

                # danger threshold - cumulative - for searcher s
                H_s = H_v[s_idx]
                md.addConstr(H_s >= X[s, v, t] * alpha)


def add_danger_constraints(md, my_vars: dict, vertices_t: dict, danger, searchers: dict, horizon: int):
    # point estimate (UT - ok)
    if danger.perception == danger.options[0]:
        list_kappa = rp.get_kappa(searchers)
        # list of estimated danger, list_hat_eta = [etahat_v1, etahat_v2, ..., etahat_vn]
        list_z_hat = danger.z_hat
        # add danger constraints
        add_kappa_point(md, my_vars, vertices_t, list_z_hat, list_kappa, horizon)

    elif danger.perception == danger.options[1]:
        list_alpha = rp.get_alpha(searchers)
        list_H = rp.get_H(danger, searchers)

        # add danger constraints
        add_kappa_prob(md, my_vars, vertices_t, list_H, list_alpha, horizon)


# -----------------------------------------------------------------------------------
# Plan --  functions
# -----------------------------------------------------------------------------------
# TODO add danger as inputs, option for point estimate or probabilistic, and call add_constraints
def run_planner(specs=None, output_data=False, printout=True):
    """Initialize the planner the pre-set parameters
        Return path of searchers as list of lists"""

    if specs is None:
        specs = rp.set_default_specs()

    belief, searchers, solver_data, target, danger = init_wrapper(specs)

    # unpack parameters
    g = specs.graph
    h = specs.horizon
    b0 = belief.new
    M = target.unpack()
    gamma = specs.gamma
    timeout = specs.timeout
    solver_type = specs.solver_type

    obj_fun, time_sol, gap, x_s, b_target, threads = run_solver(g, h, searchers, b0, M, danger, solver_type, timeout, gamma)
    searchers, path_dict = pln.update_plan(searchers, x_s)
    path_list = ext.path_as_list(path_dict)

    if output_data:
        # save the new data
        solver_data.store_new_data(obj_fun, time_sol, gap, threads, x_s, b_target, 0)

    if printout:
        pln.print_path(x_s)
        if isinstance(time_sol, dict):
            t_sol = time_sol['total']
        else:
            t_sol = time_sol
        print("Solving time: %.5f" % t_sol)

    if output_data:
        return path_list, solver_data
    else:
        return path_list


def run_solver(g, horizon, searchers, b0, M_target, danger, solver_type='central', timeout=30 * 60,
               gamma=0.99, n_inter=1, pre_solve=-1):
    """Run solver according to type of planning specified"""

    if solver_type == 'central':
        obj_fun, time_sol, gap, x_searchers, b_target, threads = central_wrapper(g, horizon, searchers, b0, M_target, danger, gamma, timeout)

    elif solver_type == 'distributed':
        obj_fun, time_sol, gap, x_searchers, b_target, threads = distributed_wrapper(g, horizon, searchers, b0, M_target, danger, gamma, timeout, n_inter, pre_solve)
    else:
        obj_fun, time_sol, gap, x_searchers, b_target, threads = mf.none_model_vars()

    return obj_fun, time_sol, gap, x_searchers, b_target, threads


# main wrappers
def central_wrapper(g, horizon, searchers, b0, M_target, danger, gamma, timeout):
    """Add variables, constraints, objective function and solve the model
    compute all paths"""

    solver_type = 'central'

    # V^{s, t}
    start, vertices_t, times_v = cm.get_vertices_and_steps(g, horizon, searchers)

    # create model
    md = mf.create_model()

    # add variables
    my_vars = mf.add_variables(md, g, horizon, start, vertices_t, searchers)

    # add constraints (central algorithm)
    mf.add_constraints(md, g, my_vars, searchers, vertices_t, horizon, b0, M_target)

    # danger constraints
    add_danger_constraints(md, my_vars, vertices_t, danger, searchers, horizon)

    # objective function
    mf.set_solver_parameters(md, gamma, horizon, my_vars, timeout)

    # solve and save results
    obj_fun, time_sol, gap, x_searchers, b_target, threads = mf.solve_model(md)

    # clean things
    md.reset()
    md.terminate()
    del md
    #

    # clean things
    return obj_fun, time_sol, gap, x_searchers, b_target, threads


def distributed_wrapper(g, horizon, searchers, b0, M_target, danger, gamma, timeout=5, n_inter=1, pre_solver=-1):
    """Distributed version of the algorithm """

    # parameter to stop iterations
    # number of full loops s= 1..m
    n_it = n_inter

    # iterative parameters
    total_time_sol = 0
    previous_obj_fun = 0
    my_counter = 0

    # temporary path for the searchers
    temp_pi = pln.init_temp_path(searchers, horizon)

    # get last searcher number [m]
    m = ext.get_last_info(searchers)[0]

    obj_fun_list = {}
    time_sol_list = {}
    gap_list = {}

    while True:

        for s_id in searchers.keys():
            # create model
            md = mf.create_model()

            temp_pi['current_searcher'] = s_id

            start, vertices_t, times_v = cm.get_vertices_and_steps_distributed(g, horizon, searchers, temp_pi)

            # add variables
            my_vars = mf.add_variables(md, g, horizon, start, vertices_t, searchers)

            mf.add_constraints(md, g, my_vars, searchers, vertices_t, horizon, b0, M_target)

            # danger constraints
            add_danger_constraints(md, my_vars, vertices_t, danger, searchers, horizon)

            # objective function
            mf.set_solver_parameters(md, gamma, horizon, my_vars, timeout, pre_solver)

            # solve and save results
            obj_fun, time_sol, gap, x_searchers, b_target, threads = mf.solve_model(md)

            if md.SolCount == 0:
                # problem was infeasible or other error (no solution found)
                print('Error, no solution found!')
                obj_fun, time_sol, gap, threads = -1, -1, -1, -1
                # keep previous belief
                b_target = {}
                v = 0
                for el in b0:
                    b_target[(v, 0)] = el
                    v += 1

                x_searchers = pln.keep_all_still(temp_pi)

            # clean things
            md.reset()
            md.terminate()
            del md

            # ------------------------------------------------------
            # append to the list
            obj_fun_list[s_id] = obj_fun
            time_sol_list[s_id] = time_sol
            gap_list[s_id] = gap

            # save the current searcher's path
            temp_pi = pln.update_temp_path(x_searchers, temp_pi, s_id)

            total_time_sol = total_time_sol + time_sol_list[s_id]

            # end of a loop through searchers
            if s_id == m:
                # compute difference between previous and current objective functions
                delta_obj = abs(previous_obj_fun - obj_fun)

                # iterate
                previous_obj_fun = obj_fun
                my_counter = my_counter + 1

                # check for stoppers: either the objective function converged or iterated as much as I wanted
                if (delta_obj < 1e-4) or (my_counter >= n_it):
                    time_sol_list['total'] = total_time_sol
                    # clean and delete
                    disposeDefaultEnv()

                    return obj_fun_list, time_sol_list, gap_list, x_searchers, b_target, threads


def init_wrapper(specs, sim=False):
    """Initialize necessary classes depending on sim or plan only
    default: plan only"""

    solver_data = rp.create_solver_data(specs)
    searchers = rp.create_searchers(specs)
    belief = cp.create_belief(specs)
    target = cp.create_target(specs)
    danger = rp.create_danger(specs)

    print('Start target: %d, searcher: %d ' % (target.current_pos, searchers[1].start))

    return belief, searchers, solver_data, target, danger


if __name__ == "__main__":

    run_planner()

    # my_specs = risk.scripts.risk_parameters.set_default_specs()
    # print(my_specs.kappa)
    # print(my_specs.eta_check)


    # f_name = 'node_score_dict_Fire.p'
    # xi = MyDanger.load_scores(f_name)
    # print('Similarity scores for node 1, image 1')
    # print(xi[1][0])
    #
    # eta, eta_hat = MyDanger.compute_frequentist(xi)
    #
    # print('\nDanger probability for node 1: ' + str(eta[1]))
    # print('Danger point estimate for node 1: level ' + str(eta_hat[1]))


