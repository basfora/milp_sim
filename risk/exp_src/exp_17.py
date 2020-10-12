"""NCDK- point - 25 and 100"""

# ---------------------------------------------------------------------------------------------------------------------
# start of header
# import relevant modules
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
specs.set_number_of_runs(1000)
# set random seeds
specs.set_start_seeds(2000, 6000)

# -----------------------
# danger specs
# ----------------------
# FILES
base_name = 'danger_map_NCF_freq_'
# true danger file
true_file = base_name + '100'
specs.set_danger_file(true_file, 'true')

# threshold of searchers
kappa = [3, 3, 3]
alpha = [0.95, 0.95, 0.95]
specs.set_threshold(kappa, 'kappa')
specs.set_threshold(alpha, 'alpha')

# danger perception
perception = 'point'
specs.set_danger_perception(perception)

# Apply danger constraints (True/False)
specs.set_danger_constraints(True)
# Apply prob kill (true/false)
specs.set_kill(True, 1)

list_img = [100, 5]

for per in list_img:
    estimated_file = base_name + str(per).zfill(2)
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


