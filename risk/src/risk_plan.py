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
from milp_sim.risk.src import base_fun as bf
from gurobipy import *


from milp_sim.risk.src import risk_param as rp
import time


# -----------------------------------------------------------------------------------
# Solver -- functions
# -----------------------------------------------------------------------------------
# UT - ok
def add_kappa_point(md, my_vars: dict, vertices_t: dict, list_z_hat: list, list_kappa: list, horizon: int):
    """
    :param md : Gurobi model
    :param horizon : planning horizon (deadline)
    :param my_vars : model variables
    :param vertices_t : allowed vertices

    :param list_z_hat : list of estimated danger,
    list_z_hat = [list_z_hat = [zhat_v1, zhat_v2, ..., zhat_vn], z_hat in L = {1, ...5}, z_hat = argmax(eta_v,l^t)

    :param list_kappa : list of danger threshold for each searcher,
    list_kappa = [k_s1, k_s2...k_sm], kappa in L = {1, 2, 3, 4, 5}

    OBS: eta is constant for planning horizon
    """

    # get variables
    X = mf.get_var(my_vars, 'x')

    m = len(list_kappa)

    S = ext.get_set_searchers(m)[0]
    T = ext.get_set_time(horizon)

    md.update()

    for s in S:
        s_idx = ext.get_python_idx(s)
        # danger threshold (kappa in L = {1,...5})
        kappa = list_kappa[s_idx]

        # 1, 2... T
        for t in T:
            v_t = vertices_t.get((s, t))

            for v in v_t:
                v_idx = ext.get_python_idx(v)
                # estimated danger level (eta_hat in L = {1,...,5})
                z_hat = list_z_hat[v_idx]
                md.addConstr(X[s, v, t] * z_hat <= kappa)


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
                md.addConstr(X[s, v, t] * alpha <= H_s)


def add_danger_constraints(md, my_vars: dict, vertices_t: dict, danger, searchers: dict, horizon: int):
    # point estimate (UT - ok)
    list_kappa = danger.kappa
    if danger.perception == danger.options[0]:

        # list of current estimated danger, list_z_hat = [zhat_v1, zhat_v2, ..., zhat_vn]
        list_z_hat = danger.z_hat
        # add danger constraints
        add_kappa_point(md, my_vars, vertices_t, list_z_hat, list_kappa, horizon)

    elif danger.perception == danger.options[1]:
        list_alpha = rp.get_alpha(searchers)
        # prep list_H with kappa
        H_levels = danger.H_hat

        list_H = danger.get_all_H_for_team(H_levels, list_kappa)
        # just to access easily
        danger.Hs_hat = list_H

        # add danger constraints
        add_kappa_prob(md, my_vars, vertices_t, list_H, list_alpha, horizon)


# -----------------------------------------------------------------------------------
# Plan --  functions
# -----------------------------------------------------------------------------------
# UT - ok
def run_planner(specs=None, sim_data=False, printout=True):
    """Initialize the planner the pre-set parameters
        Return path of searchers as list of lists"""

    if specs is None:
        specs = rp.default_specs()

    belief, team, solver_data, target, danger, mission = init_wrapper(specs)

    t = 0
    belief, target, team, solver_data, danger, inf = planner_module(belief, target, team, solver_data, danger,
                                                                    t, sim_data)
    path_list = team.get_path_list()

    if sim_data:
        return belief, target, team, solver_data, danger, inf
    else:
        return path_list


# UT - ok
def planner_module(belief, target, team, solver_data, danger, t=0, printout=True, folder_path=None):

    """Planner module to be used for planner only (sim_data=False) or simulation (sim_data = True)"""

    inf = False
    # unpack parameters
    b0 = belief.new
    g, horizon, solver_type, timeout, gamma = solver_data.unpack_for_planner()
    M = target.unpack()

    obj_fun, time_sol, gap, x_s, b_target, threads = run_solver(g, horizon, team.searchers, b0, M, danger, solver_type, timeout,
                                                                gamma)

    # break here if the problem was infeasible
    if time_sol is None or gap is None or obj_fun is None:
        inf = True
        return belief, target, team, solver_data, danger, inf

    # save the new data
    solver_data.store_new_data(obj_fun, time_sol, gap, threads, x_s, b_target, t)

    # get position of each searcher at each time-step based on x[s, v, t] variable to path [s, t] = v
    team.searchers, path_dict = pln.update_plan(team.searchers, x_s)

    path_list = ext.path_as_list(path_dict)

    if folder_path is not None:
        try:
            belief_nonzero, plan_eval, danger_ok, danger_error, my_config = check_plan(team, danger, solver_data, path_list)
            bf.save_log_file(path_list, belief_nonzero, plan_eval, danger_ok, danger_error, my_config, t, folder_path)
        except:
            pass

    if printout:
        pln.print_path_list(path_list)
        if isinstance(time_sol, dict):
            t_sol = time_sol['total']
        else:
            t_sol = time_sol
        print("Solving time: %.5f" % t_sol)

    return belief, target, team, solver_data, danger, inf


def check_plan(team, danger, solver_data, path_list: dict):
    """Check if planner is not doing anything stupid"""

    my_config = ['planner', 'priori', 'kappa', 'alpha']

    # information about danger at time t
    z_hat = danger.z_hat
    Hs_hat = danger.Hs_hat
    # problem graph
    g = solver_data.g
    V, n = ext.get_set_vertices(g)

    # all v to be visited
    vs_to_visit = []
    # sub graphs for each searcher danger level
    sub_graphs = []
    sub_Vs = []

    # ----
    # danger-aware
    # ----
    # check if all vertices are below allowed danger (given current info)
    danger_ok = True
    danger_error = ''
    # only care about the living searchers
    for s_id in team.searchers.keys():

        s = team.searchers[s_id]

        # original id (for log file only)
        s_id0 = s.id_0

        # current idx (only alive)
        s_idx = ext.get_python_idx(s_id)
        s_kappa = team.kappa[s_idx]
        s_alpha = team.alpha[s_idx]
        plan_s = path_list[s_id]

        if danger.kill and danger.constraints:
            if danger.perception == danger.options[0]:
                my_config[0] = 'PT'
                my_config[3] = 'x'
                # sub graph with vertices that danger level <= threshold
                sub_g, sub_vertices = rp.create_subgraph(g, z_hat, s_kappa)
            else:
                my_config[0] = 'PB'
                my_config[3] = str(team.alpha)
                H_s = [Hs_hat[vidx][s_idx] for vidx in range(n)]
                # sub graph with vertices that H >= alpha threshold
                sub_g, sub_vertices = rp.create_subgraph_H(g, H_s, s_alpha)

            if danger.true_priori:
                my_config[1] = 'PK'
            else:
                my_config[1] = 'PU'

            my_config[2] = str(team.kappa)

        else:
            if not danger.kill:
                my_config[0] = 'ND'
            else:
                my_config[0] = 'NC'
            my_config[1] = 'x'
            my_config[2] = 'x'
            my_config[3] = 'x'
            sub_g, sub_vertices = g, V

        # only of alive searchers
        sub_graphs.append(sub_g)
        sub_Vs.append(sub_vertices)

        for t, v in enumerate(plan_s):
            vidx = ext.get_python_idx(v)
            if v not in vs_to_visit:
                vs_to_visit.append(v)

            if t == 0:
                continue

            if danger.kill and danger.constraints:
                # get danger info for vertex v
                zhat_v = z_hat[vidx]

                # PT estimate
                if danger.perception == danger.options[0] and zhat_v > s_kappa:
                    danger_error = 'Error in planned path! s = ' + str(s_id0) + ' k = ' + str(s_kappa) + \
                                   ' || v = ' + str(v) + ' z_hat = ' + str(zhat_v)
                    print(danger_error)
                    danger_ok = False

                # PB estimate
                elif danger.perception == danger.options[1]:
                    Hs_hat_v = Hs_hat[vidx][s_idx]
                    if Hs_hat_v < s_alpha:
                        danger_error = 'Error in planned path! s = ' + str(s_id0) + ' k = ' + str(s_alpha) + \
                                    ' || v = ' + str(v) + ' z_hat = ' + str(Hs_hat_v)
                        print(danger_error)
                        danger_ok = False

    belief_nonzero, plan_eval = check_wrt_belief(solver_data, team, path_list, sub_graphs, sub_Vs, vs_to_visit)

    return belief_nonzero, plan_eval, danger_ok, danger_error, my_config


def check_wrt_belief(solver_data, team, path_list, sub_graphs, sub_Vs, vs_to_visit):

    # tic = time.perf_counter()

    # planning horizon
    h = solver_data.horizon[0]
    # belief vector computed by the planner at t = 0
    t = 0
    belief_nonzero = solver_data.get_nonzero_beta(t)

    # ----
    # target capture
    # ----
    plan_eval = []

    # for each vertex with some prob of finding the target
    for vb in belief_nonzero:

        # just to debug, (1, [2], [3], [4])
        reward_collect = False
        # none of the options above and vertex not being considered (5)
        reason = []

        # (1) reward is being collected
        # check if vb will be visited (by any searcher):
        if vb in vs_to_visit:
            reward_collect = True
            plan_eval.append((reward_collect, reason))
            # move on to next vertex in belief
            continue

        # if not, let's find out why
        # go through each searcher allowed vertices
        s_id = 0
        for s_id0 in team.searchers_original.keys():

            if s_id0 not in team.alive:
                reason.append(-1)
                continue

            s_id += 1
            too_dangerous, not_connected, too_far, current_pos = False, False, False, False
            s_idx = ext.get_python_idx(s_id)
            # sub graph
            sub_g = sub_graphs[s_idx]
            allowed_vertices = sub_Vs[s_idx]
            # searcher current position
            s_pos = path_list[s_id][0]

            # maybe it's not reachable
            # (2) too dangerous: vb is not part of searcher's subgraph
            if vb not in allowed_vertices:
                too_dangerous = True

            # if it's in the graph, it might be unconnected or too far from current searcher position
            else:

                if s_pos not in allowed_vertices:
                    connection, distance = rp.connected(solver_data.g, s_pos, vb)
                    current_pos = True
                else:
                    connection, distance = rp.connected(sub_g, s_pos, vb)

                # is connected
                if connection:
                    # (4) but too far
                    if distance > h:
                        too_far = True
                else:
                    # (3) not connected
                    not_connected = True

            # store reason
            if too_dangerous:
                reason.append(2)
            elif not_connected:
                reason.append(3)
            elif too_far:
                reason.append(4)
            elif current_pos:
                reason.append(5)
            else:
                reason.append(None)

        plan_eval.append((reward_collect, reason))

    # toc = time.perf_counter()
    # print('Time = %s' % str(toc-tic))

    return belief_nonzero, plan_eval


def print_plan_eval(belief_nonzero, plan_eval):

    for i, vb in enumerate(belief_nonzero):
        print('Vertex %d' % vb)

        if plan_eval[i][0]:
            print('will be visited')
        else:
            print('not in plan because %s' % str(plan_eval[i][1]))

    return


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

    # apply danger constraints
    if danger.constraints:
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

            if danger.constraints:
                # apply danger constraints
                add_danger_constraints(md, my_vars, vertices_t, danger, searchers, horizon)

            # objective function
            mf.set_solver_parameters(md, gamma, horizon, my_vars, timeout, pre_solver)

            # solve and save results
            obj_fun, time_sol, gap, x_searchers, b_target, threads = mf.solve_model(md)

            if md.SolCount == 0:
                # problem was infeasible or other error (no solution found)

                print('x-------------------------------------------------x\nError, no solution found!')
                v_s = start[s_id-1]
                print('Computing path for s = %d, kappa = %d, at v = %d' % (s_id, searchers[s_id].kappa, v_s))
                print('z_hat = %s' % str(danger.z_hat[v_s-1]))
                print('eta_hat = %s' % str(danger.eta_hat[v_s-1]))
                list_H = danger.compute_all_H(danger.eta_hat)
                print('H_hat = %s \nx-------------------------------------------------x' % str(list_H[v_s-1]))

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


# UT - ok
def init_wrapper(specs):
    """Initialize necessary classes depending on sim or plan only
    default: plan only"""

    solver_data = rp.create_solver_data(specs)
    team = rp.create_team(specs)
    belief = cp.create_belief(specs)
    target = cp.create_target(specs)
    danger = rp.create_danger(specs)
    mission = rp.create_mission(specs)

    print('Start target: %d, searcher: %d ' % (target.current_pos, team.searchers[1].start))

    return belief, team, solver_data, target, danger, mission


if __name__ == "__main__":
    # run planner with default specs
    run_planner()



