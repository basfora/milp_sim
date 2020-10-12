"""Experiment with fake data just for testing of point estimate"""

# ---------------------------------------------------------------------------------------------------------------------
# start of header
# import relevant modules
import risk.classes.child_mespp
from milp_sim.risk.classes.child_mespp import MyInputs2
from milp_sim.risk.src import sim_risk as sr, risk_parameters as rp, base_fun as bf
from milp_mespp.core import plot_fun as pf
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

# danger specs
# true danger file
true_file = 'danger_map_freq_100'
# estimated_file = 'danger_map_25'
specs.set_danger_file(true_file, 'true')
# estimated danger file
# specs.set_danger_file(estimated_file, 'hat')
# threshold of searchers
kappa = [3, 4, 5]
alpha = [0.95, 0.95, 0.95]
specs.set_threshold(kappa, 'kappa')
specs.set_threshold(alpha, 'alpha')
# danger perception
perception = 'point'
specs.set_danger_perception(perception)

list_per = [25, 50, 75, 100]

for per in list_per:
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
        del belief, target, team, solver_data, danger, mission
        print("----------------------------------------------------------------------------------------------------")
        print("----------------------------------------------------------------------------------------------------")


