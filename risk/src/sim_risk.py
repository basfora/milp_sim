from milp_sim.risk.src import plan_risk as plnr, risk_parameters as rp
from milp_mespp.core import plan_fun as pln
from milp_mespp.core import sim_fun as sf, extract_info as ext


def run_simulator(specs=None, kill=True):
    """Easy handle to run simulator"""

    if specs is None:
        specs = rp.default_specs()

    belief, target, team, solver_data, danger = risk_simulator(specs, kill)

    return belief, target, team, solver_data, danger


def risk_simulator(specs, kill=True, printout=True):
    """ Main risk simulator function
    Input: specs from MyInputs()
    Return: belief, searchers, solver_data, target, danger"""

    # extract inputs for the problem instance
    # timeout = specs.timeout
    # g = specs.graph

    # initialize classes
    belief, team, solver_data, target, danger = plnr.init_wrapper(specs)
    # -------------------------------------------------------------------------------

    deadline, theta, n = solver_data.unpack_for_sim()
    # deadline, horizon, theta, solver_type, gamma = solver_data.unpack()
    M = target.unpack()

    # initialize time: actual sim time, t = 0, 1, .... T and time relative to the planning, t_idx = 0, 1, ... H
    t, t_plan = 0, 0
    path = {}
    # _____________________

    # begin simulation loop
    while t < deadline:

        print('--\nTime step %d \n--' % t)

        # _________________
        # check if it's time to re-plan (OBS: it will plan on t = 0)
        if t % theta == 0:

            # call for planner module
            sim_data = True
            belief, target, team, solver_data, danger, inf = plnr.planner_module(belief, target, team, solver_data,
                                                                                 danger, t, sim_data)

            # break here if the problem was infeasible
            if inf:
                break

            # get planned path as dict:  path [s, t] = v
            path = team.get_path()

            # reset time-steps of planning
            t_plan = 1

        # _________________

        if printout:
            # print current positions
            print('t = %d' % t)
            sf.print_positions(team.searchers, target)

        # get dictionary of next positions for searchers, new_pos = {s: v}
        path_next_t = pln.next_from_path(path, t_plan)

        # evolve searcher positions
        team.searchers_evolve(path_next_t)

        # --------------------------------------
        # new [danger]
        # estimate danger
        danger.estimate(team.visited_vertices)

        if kill:
            # compute prob kill, draw your luck and update searchers (if needed)
            team.decide_searchers_luck(danger, t)
            # update info in danger
            danger.update_teamsize(len(team.alive))
            danger.set_thresholds(team.kappa, team.alpha)

        # retrieve path_next_t (from searchers that are still alive)
        path_next_t = team.retrieve_current_positions()
        # --------------------------------------

        # update belief
        belief.update(team.searchers, path_next_t, M, n)

        # update target
        target = sf.evolve_target(target, belief.new)

        # next time-step
        t, t_plan = t + 1, t_plan + 1

        # --------------------------------------
        # new [check if searchers are alive]
        if len(team.alive) < 1:
            print('Mission failed, all searchers were killed.')
            break
        # --------------------------------------

        # check for capture based on next position of vertex and searchers
        team.searchers, target = sf.check_for_capture(team.searchers, target)

        if (t == deadline) and printout:
            print('--\nTime step %d\n--' % deadline)
            print('t = %d' % t)
            sf.print_positions(team.searchers, target)

            if target.is_captured is False:
                print('Mission failed, target was not found within deadline.')

        if target.is_captured:
            sf.print_capture_details(t, target, team.searchers, solver_data)
            break

    return belief, target, team, solver_data, danger



