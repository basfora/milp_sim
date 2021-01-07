"""Standard experiments for RAL 2021"""
from milp_sim.risk.classes.child_mespp import MyInputs2
from milp_sim.risk.src import base_fun as bf, risk_sim as sr


def specs_basic():
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
    specs.set_capture_range(0)
    specs.set_zeta(None)
    # ------------------------
    # target motion
    specs.set_target_motion('static')
    specs.set_start_target_vertex(None)

    return specs


def specs_num_sim():
    """Set specs that won't change"""

    specs = specs_basic()

    # ------------------------
    # searchers' detection: capture range and false negatives
    m = 3
    v0 = [1, 1, 1]
    specs.set_size_team(m)
    specs.set_start_searchers(v0)

    # ------------------------
    # pseudorandom, repetitions for each configuration
    specs.set_number_of_runs(1000)
    # set random seeds
    specs.set_start_seeds(2000, 6000)

    return specs


def default_thresholds():
    # threshold of searchers
    kappa = [3, 4, 5]
    alpha = [0.6, 0.4, 0.4]

    return kappa, alpha


# ------------------------------------------
""" Default specs danger
Planner: point || Priori danger knowledge: uniform || Team makeup: k = [3, 4, 5]"""


# PT-PU-345
def specs_danger_common():
    """Set common danger specs
    """

    specs = specs_num_sim()

    # danger files
    base_name = 'estimate_danger_fire_des_NFF_freq_'
    human_gt = 'gt_danger_NFF'
    # true danger file
    true_file = human_gt
    specs.set_danger_file(true_file, 'true')
    # ----------------------------------------
    # estimating danger with 5% images
    # ----------------------------------------
    per = 5
    estimated_file = base_name + str(per).zfill(2)
    # estimated danger file
    specs.set_danger_file(estimated_file, 'hat')

    kappa, alpha = default_thresholds()

    specs.set_threshold(kappa, 'kappa')
    specs.set_threshold(alpha, 'alpha')

    # danger perception - point is default
    perception = 'point'
    specs.set_danger_perception(perception)

    # Apply prob kill (true/false)
    # hybrid prob (op 3) - default, do not change
    default_prob = 3
    # apply p(loss|lv)
    specs.use_kill(True, default_prob)
    # compute z as max value, break ties as K_mva + 1 (if necessary)
    specs.set_mva_conservative(True)
    # use FOV in school
    specs.set_use_fov(True)
    # a priori prob -> uniform
    specs.set_true_estimate(False)

    return specs


# change planner
def specs_prob(specs=None):

    if specs is None:
        specs = specs_danger_common()

    # danger perception
    perception = 'prob'
    specs.set_danger_perception(perception)

    return specs


# ------------------------------------------
""" No constraints enforced"""
# ------------------------------------------


# ND
def specs_no_danger(specs_in=None):
    """ND: best case scenario, no p(loss|lv)"""

    if specs_in is None:
        specs_in = specs_danger_common()

    specs_in.use_kill(False)
    specs_in.use_danger_constraints(False)

    specs = specs_in

    return specs


# NC
def specs_no_constraints(specs_in=None):
    """NC: worst case, has p(loss|lv) but no danger constraints"""

    if specs_in is None:
        specs_in = specs_danger_common()

    specs_in.use_kill(True, 3)
    specs_in.use_danger_constraints(False)

    return specs_in


# ------------------------------------------
""" Change specs"""


# change team makeup
def kappa_335(specs=None):

    if specs is None:
        specs = specs_danger_common()

    kappa = [3, 3, 5]
    alpha = [0.6, 0.6, 0.4]
    specs.set_threshold(kappa, 'kappa')
    specs.set_threshold(alpha, 'alpha')

    return specs


def kappa_333(specs=None):

    if specs is None:
        specs = specs_danger_common()

    kappa = [3, 3, 3]
    alpha = [0.6, 0.6, 0.6]
    specs.set_threshold(kappa, 'kappa')
    specs.set_threshold(alpha, 'alpha')

    return specs


# change a priori
def uniform_priori(specs=None):
    if specs is None:
        specs = specs_danger_common()

    # a priori prob -> uniform
    specs.set_true_estimate(False)

    return specs


def perfect_priori(specs=None):

    if specs is None:
        specs = specs_danger_common()

    # set perfect a priori knowledge
    specs.set_true_know(True)
    # and estimate
    specs.set_true_estimate(True)

    return specs


# ------------------------------------------
"""Actual configs"""
# ------------------------------------------


def pt_pu_345(specs_in=None):

    if specs_in is None:
        specs_in = specs_danger_common()

    specs = specs_in

    return specs


def pt_pu_335(specs_in=None):
    """Alternative team makeup, point estimate, uniform a priori | PT-PU-335"""
    if specs_in is None:
        specs_in = specs_danger_common()
    specs = kappa_335(specs_in)

    return specs


def pt_pu_333(specs_in=None):
    """Alternative team makeup, point estimate, uniform a priori | PT-PU-333"""
    if specs_in is None:
        specs_in = specs_danger_common()
    specs = kappa_333(specs_in)

    return specs


def pt_pk_345(specs_in=None):
    if specs_in is None:
        specs_in = specs_danger_common()
    specs = perfect_priori(specs_in)

    return specs


def pt_pk_335(specs_in=None):
    if specs_in is None:
        specs_in = specs_danger_common()

    specs_345 = perfect_priori(specs_in)
    specs = kappa_335(specs_345)

    return specs


def pt_pk_333(specs_in=None):
    if specs_in is None:
        specs_in = specs_danger_common()
    specs_345 = perfect_priori(specs_in)
    specs = kappa_335(specs_345)

    return specs


# ------------------------------------------
""" Planner: cumulative probability"""
# ------------------------------------------


def pb_pu_345(specs_in=None):
    specs_pb = specs_prob(specs_in)

    return specs_pb


def pb_pu_335(specs_in=None):
    specs_pb = specs_prob(specs_in)
    specs = kappa_335(specs_pb)

    return specs


def pb_pu_333(specs_in=None):
    """Alternative team makeup, point estimate, uniform a priori | PT-PU-333"""
    specs_pb = specs_prob(specs_in)
    specs = kappa_333(specs_pb)

    return specs


def pb_pk_345(specs_in=None):
    specs_pb = specs_prob(specs_in)
    specs = perfect_priori(specs_pb)

    return specs


def pb_pk_335(specs_in=None):
    specs_pb = specs_prob(specs_in)
    specs_345 = perfect_priori(specs_pb)
    specs = kappa_335(specs_345)

    return specs


def pb_pk_333(specs_in=None):
    specs_pb = specs_prob(specs_in)
    specs_345 = perfect_priori(specs_pb)
    specs = kappa_335(specs_345)

    return specs


def get_believes():
    specs = specs_num_sim()

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

    specs.use_log_file()

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








