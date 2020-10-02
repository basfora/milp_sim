from milp_sim.risk.src import plan_risk as plnr, risk_parameters as rp
from milp_mespp.core import plan_fun as pln
from milp_mespp.core import sim_fun as sf, extract_info as ext


def run_simulator(specs=None):
    """Easy handle to run simulator"""

    if specs is None:
        specs = rp.default_specs()

    belief, target, searchers, solver_data, danger = risk_simulator(specs)

    return belief, target, searchers, solver_data, danger


def risk_simulator(specs, printout=True):
    """ Main risk simulator function
    Input: specs from MyInputs()
    Return: belief, searchers, solver_data, target, danger"""

    # extract inputs for the problem instance
    timeout = specs.timeout
    g = specs.graph
    m = specs.size_team

    # initialize classes
    belief, searchers, solver_data, target, danger = plnr.init_wrapper(specs, True)
    # -------------------------------------------------------------------------------

    deadline, horizon, theta, solver_type, gamma = solver_data.unpack()
    M = target.unpack()

    # get sets for easy iteration
    S, V, _, m, n = ext.get_sets_and_ranges(g, m, horizon)

    # initialize time: actual sim time, t = 0, 1, .... T and time relative to the planning, t_idx = 0, 1, ... H
    t, t_plan = 0, 0
    path = {}
    # _____________________

    # begin simulation loop
    while t < deadline:

        print('--\nTime step %d \n--' % t)

        # _________________
        if t % theta == 0:
            # check if it's time to re-plan (OBS: it will plan on t = 0)

            # call for model solver wrapper according to centralized or decentralized solver and return the solver data
            obj_fun, time_sol, gap, x_s, b_target, threads = plnr.run_solver(g, horizon, searchers, belief.new, M,
                                                                             danger, solver_type, timeout, gamma)

            # save the new data
            solver_data.store_new_data(obj_fun, time_sol, gap, threads, x_s, b_target, t)

            # break here if the problem was infeasible
            if time_sol is None or gap is None or obj_fun is None:
                break

            # get position of each searcher at each time-step based on x[s, v, t] variable to path [s, t] = v
            searchers, path = pln.update_plan(searchers, x_s)

            if printout:
                pln.print_path(x_s)

            # reset time-steps of planning
            t_plan = 1

        # _________________

        if printout:
            # print current positions
            print('t = %d' % t)
            sf.print_positions(searchers, target)

        # get dictionary of next positions for searchers, new_pos = {s: v}
        path_next_t = pln.next_from_path(path, t_plan)

        # evolve searcher position
        searchers = pln.searchers_evolve(searchers, path_next_t)

        # update belief
        belief.update(searchers, path_next_t, M, n)

        # update target
        target = sf.evolve_target(target, belief.new)

        # next time-step
        t, t_plan = t + 1, t_plan + 1

        # check for capture based on next position of vertex and searchers
        searchers, target = sf.check_for_capture(searchers, target)

        if (t == deadline) and printout:
            print('--\nTime step %d\n--' % deadline)
            print('t = %d' % t)
            sf.print_positions(searchers, target)

        if target.is_captured:
            sf.print_capture_details(t, target, searchers, solver_data)
            break

    return belief, target, searchers, solver_data, danger
