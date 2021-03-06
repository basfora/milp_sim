"""Standard experiments for ICRA 2021"""
from milp_sim.risk.classes.child_mespp import MyInputs2
from milp_sim.risk.src import base_fun as bf, risk_sim as sr


def specs_basic():
    """Set specs that won't change"""

    # initialize default inputs
    specs = MyInputs2()
    # ------------------------
    # graph number -- SS-2: 8
    specs.set_graph(8)
    # solver parameter: central x distributed
    specs.set_solver_type('distributed')
    # solver timeout (in seconds)
    specs.set_timeout(10)
    # ------------------------
    # time stuff: deadline mission (tau), planning horizon (h), re-plan frequency (theta)
    specs.set_horizon(14)
    specs.set_deadline(100)
    specs.set_theta(1)
    # ------------------------
    # searchers' detection: capture range and false negatives
    m = 3
    v0 = [1, 1, 1]
    specs.set_size_team(m)
    specs.set_start_searchers(v0)
    specs.set_capture_range(0)
    specs.set_zeta(None)
    # ------------------------
    # target motion
    specs.set_target_motion('static')
    specs.set_start_target_vertex(None)
    # ------------------------
    # pseudorandom
    # repetitions for each configuration
    specs.set_number_of_runs(1000)
    # set random seeds
    specs.set_start_seeds(2000, 6000)

    return specs


def specs_danger_common():
    """Set common danger specs
    """
    specs = specs_basic()

    # danger files
    base_name = 'danger_map_NCF_freq_'
    # true danger file
    true_file = base_name + '100'
    specs.set_danger_file(true_file, 'true')
    # ----------------------------------------
    # estimating danger with 5% images
    # ----------------------------------------
    per = 5
    estimated_file = base_name + str(per).zfill(2)
    # estimated danger file
    specs.set_danger_file(estimated_file, 'hat')

    # threshold of searchers
    kappa = [3, 4, 5]
    alpha = [0.95, 0.95, 0.95]
    specs.set_threshold(kappa, 'kappa')
    specs.set_threshold(alpha, 'alpha')

    # danger perception
    perception = 'point'
    specs.set_danger_perception(perception)

    # Apply prob kill (true/false)
    # hybrid prob (op 3)
    default_prob = 3
    specs.use_kill(True, default_prob)
    specs.set_mva_conservative(True)
    specs.set_use_fov(True)
    specs.set_true_estimate(False)

    return specs


# ------------------------------------------
def specs_true_priori():
    specs = specs_danger_common()
    # set perfect a priori knowledge
    specs.set_true_know(True)
    # and estimate
    specs.set_true_estimate(True)

    return specs


def specs_no_danger():
    specs = specs_danger_common()
    specs.use_kill(False)
    specs.use_danger_constraints(False)

    return specs


def specs_no_constraints():
    specs = specs_danger_common()

    specs.use_kill(True, 3)
    specs.use_danger_constraints(False)

    return specs


def specs_no_fov():
    specs = specs_danger_common()

    specs.set_use_fov(False)

    return specs


def specs_100_img():

    specs = specs_danger_common()

    # danger files
    base_name = 'danger_map_NCF_freq_'
    # true danger file
    true_file = base_name + '100'
    specs.set_danger_file(true_file, 'true')
    # ----------------------------------------
    # estimating only with 5% images
    # ----------------------------------------
    per = 100
    estimated_file = base_name + str(per).zfill(2)
    # estimated danger file
    specs.set_danger_file(estimated_file, 'hat')

    return specs


def specs_335():
    specs = specs_danger_common()

    kappa = [3, 3, 5]
    specs.set_threshold(kappa, 'kappa')

    return specs


def specs_333():
    specs = specs_danger_common()

    kappa = [3, 3, 3]
    specs.set_threshold(kappa, 'kappa')

    return specs


# ------------------------------------------
def specs_prob():
    specs = specs_danger_common()
    # threshold of searchers
    kappa = [3, 4, 5]
    alpha = [0.6, 0.4, 0.4]
    specs.set_threshold(kappa, 'kappa')
    specs.set_threshold(alpha, 'alpha')

    # danger perception
    perception = 'prob'
    specs.set_danger_perception(perception)

    return specs


def specs_true_priori_prob():
    specs = specs_prob()

    # set perfect a priori knowledge
    specs.set_true_know(True)
    # and estimate
    specs.set_true_estimate(True)

    return specs


def specs_no_fov_prob():
    specs = specs_prob()

    specs.set_use_fov(False)

    return specs


def specs_100_img_prob():

    specs = specs_prob()

    # danger files
    base_name = 'danger_map_NCF_freq_'
    # true danger file
    true_file = base_name + '100'
    specs.set_danger_file(true_file, 'true')
    # ----------------------------------------
    # estimating only with 5% images
    # ----------------------------------------
    per = 100
    estimated_file = base_name + str(per).zfill(2)
    # estimated danger file
    specs.set_danger_file(estimated_file, 'hat')

    return specs


def specs_335_prob():
    specs = specs_prob()

    kappa = [3, 3, 5]
    alpha = [0.6, 0.6, 0.1]
    specs.set_threshold(kappa, 'kappa')
    specs.set_threshold(alpha, 'alpha')

    return specs


def specs_NFF_both():
    specs = specs_basic()
    # danger files
    # true danger file
    true_file = 'gt_danger_NFF'
    specs.set_danger_file(true_file, 'true')

    # ----------------------------------------
    # estimating danger with 5% images
    # ----------------------------------------
    per = 5
    estimated_file = 'estimate_danger_both_des_NFF_freq_05'
    # estimated danger file
    specs.set_danger_file(estimated_file, 'hat')
    specs.set_gt(True)
    specs.set_all_descriptions(True)

    # threshold of searchers
    kappa = [3, 4, 5]
    alpha = [0.6, 0.4, 0.4]
    specs.set_threshold(kappa, 'kappa')
    specs.set_threshold(alpha, 'alpha')

    # danger perception
    perception = 'point'
    specs.set_danger_perception(perception)

    # Apply prob kill (true/false)
    # hybrid prob (op 3)
    default_prob = 3
    specs.use_kill(True, default_prob)
    specs.set_mva_conservative(True)
    specs.set_use_fov(True)
    specs.set_true_estimate(False)

    return specs


def specs_NFF_both_est():
    specs = specs_basic()
    # danger files
    # true danger file
    true_file = 'estimate_danger_both_des_NFF_freq_100'
    specs.set_danger_file(true_file, 'true')
    # ----------------------------------------
    # estimating danger with 5% images
    # ----------------------------------------
    per = 5
    estimated_file = 'estimate_danger_both_des_NFF_freq_05'
    # estimated danger file
    specs.set_danger_file(estimated_file, 'hat')

    specs.set_gt(False)
    specs.set_all_descriptions(True)

    # threshold of searchers
    kappa = [3, 4, 5]
    alpha = [0.6, 0.4, 0.4]
    specs.set_threshold(kappa, 'kappa')
    specs.set_threshold(alpha, 'alpha')

    # danger perception
    perception = 'point'
    specs.set_danger_perception(perception)

    # Apply prob kill (true/false)
    # hybrid prob (op 3)
    default_prob = 3
    specs.use_kill(True, default_prob)
    specs.set_mva_conservative(True)
    specs.set_use_fov(True)
    specs.set_true_estimate(False)

    return specs


def specs_NFF_fire():
    specs = specs_basic()
    # danger files
    # true danger file
    true_file = 'gt_danger_NFF'
    specs.set_danger_file(true_file, 'true')
    # ----------------------------------------
    # estimating danger with 5% images
    # ----------------------------------------
    per = 5
    estimated_file = 'estimate_danger_fire_des_NFF_freq_05'
    # estimated danger file
    specs.set_danger_file(estimated_file, 'hat')

    specs.set_gt(True)
    specs.set_all_descriptions(False)

    # threshold of searchers
    kappa = [3, 4, 5]
    alpha = [0.6, 0.4, 0.4]
    specs.set_threshold(kappa, 'kappa')
    specs.set_threshold(alpha, 'alpha')

    # danger perception
    perception = 'point'
    specs.set_danger_perception(perception)

    # Apply prob kill (true/false)
    # hybrid prob (op 3)
    default_prob = 3
    specs.use_kill(True, default_prob)
    specs.set_mva_conservative(True)
    specs.set_use_fov(True)
    specs.set_true_estimate(False)

    return specs


def specs_NFF_fire_est():
    specs = specs_basic()
    # danger files
    # true danger file
    true_file = 'estimate_danger_fire_des_NFF_freq_100'
    specs.set_danger_file(true_file, 'true')
    # ----------------------------------------
    # estimating danger with 5% images
    # ----------------------------------------
    per = 5
    estimated_file = 'estimate_danger_fire_des_NFF_freq_05'
    # estimated danger file
    specs.set_danger_file(estimated_file, 'hat')

    specs.set_gt(False)
    specs.set_all_descriptions(False)

    # threshold of searchers
    kappa = [3, 4, 5]
    alpha = [0.6, 0.4, 0.4]
    specs.set_threshold(kappa, 'kappa')
    specs.set_threshold(alpha, 'alpha')

    # danger perception
    perception = 'point'
    specs.set_danger_perception(perception)

    # Apply prob kill (true/false)
    # hybrid prob (op 3)
    default_prob = 3
    specs.use_kill(True, default_prob)
    specs.set_mva_conservative(True)
    specs.set_use_fov(True)
    specs.set_true_estimate(False)

    return specs


def specs_new_true_335():
    specs = specs_NFF_both()

    # threshold of searchers
    kappa = [3, 3, 5]
    alpha = [0.6, 0.6, 0.4]
    specs.set_threshold(kappa, 'kappa')
    specs.set_threshold(alpha, 'alpha')


def get_believes():
    specs = specs_basic()

    for turn in specs.list_turns:

        # set seed according to run #
        specs.set_seeds(turn)
        # set new belief
        n = 46
        b0 = MyInputs2.pick_random_belief(n, specs.target_seed)

        # print(b0)

        if turn > 30:
            break


# ------------------------------------------


def num_sim(specs):
    # loop for number of repetitions
    for turn in specs.list_turns:

        specs.prep_next_turn(turn)

        # try:

        # run simulator
        belief, target, team, solver_data, danger, mission = sr.run_simulator(specs)

        # save everything as a pickle file
        bf.save_sim_data(belief, target, team, solver_data, danger, specs, mission)

        # iterate run #
        specs.update_run_number()

        # delete things
        del belief, target, team, solver_data, danger, mission
        #except:
         #   print('Error on instance %d! Jumping to next instance.' % turn)
            # iterate run #
         #   specs.update_run_number()
         #   pass

        print("----------------------------------------------------------------------------------------------------")
        print("----------------------------------------------------------------------------------------------------")















