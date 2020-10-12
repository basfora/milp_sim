"""Experiment with fake data just for testing of point estimate"""

# ---------------------------------------------------------------------------------------------------------------------
# start of header
# import relevant modules
import risk.classes.child_mespp
from milp_sim.risk.classes.child_mespp import MyInputs2
from milp_sim.risk.src import sim_risk as sr, risk_parameters as rp, base_fun as bf
from milp_mespp.core import plot_fun as pf
from math import exp
# end of header
# ---------------------------------------------------------------------------------------------------------------------
# initialize default inputs
specs = MyInputs2()
# graph number -- SS-2: 8
specs.set_graph(8)
# solver parameter: central x distributed
specs.set_solver_type('distributed')
# time stuff: deadline mission (tau), planning horizon (h), re-plan frequency (theta)
specs.set_horizon(14)
specs.set_deadline(100)
specs.set_theta(1)
# solver timeout (in seconds)
specs.set_timeout(10)

# searchers' detection: capture range and false negatives
m = 3
v0 = [1, 1, 1]
specs.set_size_team(m)
specs.set_start_searchers(v0)
specs.set_capture_range(0)
specs.set_zeta(None)

# target motion
specs.set_target_motion('static')
specs.set_start_target_vertex(None)

# repetitions for each configuration
specs.set_number_of_runs(200)
# set random seeds
specs.set_start_seeds(2000, 6000)

# -----------------------
# danger specs
# ----------------------
# FILES
# true danger file
true_file = 'danger_map_freq_100'
# estimated_file = 'danger_map_25'
specs.set_danger_file(true_file, 'true')
# estimated danger file
# specs.set_danger_file(estimated_file, 'hat')

# levels
levels = [1, 2, 3, 4, 5]
prob_list = [0.0035 * exp(level) for level in levels]
specs.set_prob_kill(prob_list)

# threshold of searchers
kappa = [3, 3, 3]
alpha = [0.95, 0.95, 0.95]
specs.set_threshold(kappa, 'kappa')
specs.set_threshold(alpha, 'alpha')

# danger perception
perception = 'point'
specs.set_danger_perception(perception)

# Worst-case: no danger, no constraints
specs.set_danger_constraints(False)
specs.set_kill(True)
specs.set_homo(True)

list_img = [25, 50, 75, 100]

for per in list_img:
    estimated_file = 'danger_map_freq_' + str(per)
    # estimated danger file
    specs.set_danger_file(estimated_file, 'hat')

    # loop for number of repetitions
    for turn in specs.list_turns:

        specs.prep_next_turn(turn)

        # run simulator
        belief, target, team, solver_data, danger, mission = sr.run_simulator(specs)

        # save everything as a pickle file
        bf.save_sim_data(belief, target, team, solver_data, danger, specs, mission)

        # iterate run #
        today_run = specs.update_run_number()

        if turn < 0:
            # if wanting to plot
            pf.plot_sim_results(belief, target, team.searchers, solver_data, specs.name_folder)

        # delete things
        del belief, target, team, solver_data, danger
        print("----------------------------------------------------------------------------------------------------")
        print("----------------------------------------------------------------------------------------------------")


