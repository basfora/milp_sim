"""Test if ral_default is setting the right things"""

from milp_sim.risk.exp_src import ral_default as ral
from milp_sim.risk.classes import gazebo as gzb


def test_basic_specs():
    specs = ral.specs_basic()

    assert specs.solver_type == 'distributed'

    assert specs.timeout == 10
    assert specs.horizon == 14
    assert specs.deadline == 100
    assert specs.theta == 1
    assert specs.capture_range == 0
    assert specs.zeta is None
    assert specs.target_motion == 'static'


def test_default_thresholds():
    kappa, alpha = ral.default_thresholds()

    assert kappa == [3, 4, 5]
    assert alpha == [0.6, 0.4, 0.4]


def test_common_danger():
    specs = ral.specs_danger()

    # check specs basic ok
    assert specs.horizon == 14
    # check if danger was set correctly
    assert specs.danger_true == 'gt_danger_NFF'

    assert specs.kappa == [3, 4, 5]
    assert specs.alpha == [0.6, 0.4, 0.4]
    assert specs.danger_kill is True
    assert specs.danger_constraints is True


def test_num_sim():

    specs = ral.specs_num_sim()

    # check specs basic ok
    assert specs.horizon == 14
    # check if danger was set correctly
    assert specs.danger_true == 'gt_danger_NFF'
    # check team size, start for num sim ok
    assert specs.size_team == 3
    assert specs.start_searcher_v == [1, 1, 1]

    # check number of runs
    assert specs.runs_per_m == 1000


def test_gazebo_sim():

    # original number of searchers
    m = 3
    specs = ral.specs_gazebo_sim(m)

    # danger
    assert specs.kappa == [3, 4, 5]
    assert specs.alpha == [0.6, 0.4, 0.4]
    # change in size
    assert specs.size_team == 3
    # check runs
    assert specs.runs_per_m == 1

    # check specs basic ok
    assert specs.horizon == 14
    # check if danger was set correctly
    assert specs.danger_true == 'gt_danger_NFF'

    m = 2
    specs = ral.specs_gazebo_sim(m)

    # check specs basic ok
    assert specs.horizon == 14
    # check if danger was set correctly
    assert specs.danger_true == 'gt_danger_NFF'
    assert specs.kappa == [3, 4, 5]
    assert specs.alpha == [0.6, 0.4, 0.4]

    # change in size
    assert specs.size_team == 2
    # check runs
    assert specs.runs_per_m == 1


def test_no_danger():
    """sim op 4"""

    # no input
    specs = ral.specs_no_danger()

    assert specs.danger_kill is False
    assert specs.danger_constraints is False

    # default thresholds are still stored there
    assert specs.kappa == [3, 4, 5]
    assert specs.alpha == [0.6, 0.4, 0.4]
    # check specs basic ok
    assert specs.horizon == 14
    # check if danger was set correctly
    assert specs.danger_true == 'gt_danger_NFF'

    # with input
    specs2 = ral.specs_gazebo_sim(2)

    specs = ral.specs_no_danger(specs2)

    # check specs basic ok
    assert specs.horizon == 14
    # check if danger was set correctly
    assert specs.danger_true == 'gt_danger_NFF'

    assert specs.size_team == 2


def test_prob():
    # no input
    specs = ral.specs_prob()

    # check specs basic ok
    assert specs.horizon == 14
    # check if danger was set correctly
    assert specs.danger_true == 'gt_danger_NFF'

    assert specs.perception == specs.perception_list[1]

    # with input
    specs2 = ral.specs_gazebo_sim(2)
    specs = ral.specs_prob(specs2)

    # check specs basic ok
    assert specs.horizon == 14
    # check if danger was set correctly
    assert specs.danger_true == 'gt_danger_NFF'
    assert specs.perception == specs.perception_list[1]
    assert specs.size_team == 2


def test_pt_pu_345():
    # no input
    specs = ral.pt_pu_345()

    # check specs basic ok
    assert specs.horizon == 14
    # check if danger was set correctly
    assert specs.danger_true == 'gt_danger_NFF'
    assert specs.perception == specs.perception_list[0]
    assert specs.kappa == [3, 4, 5]

    # with input
    specs2 = ral.specs_gazebo_sim(2)
    specs = ral.pt_pu_345(specs2)

    # check specs basic ok
    assert specs.horizon == 14
    # check if danger was set correctly
    assert specs.danger_true == 'gt_danger_NFF'
    assert specs.perception == specs.perception_list[0]
    assert specs.size_team == 2
    assert specs.kappa == [3, 4, 5]


def test_pt_pk_345():
    # no input
    specs = ral.pt_pk_345()

    # check specs basic ok
    assert specs.horizon == 14
    # check if danger was set correctly
    assert specs.danger_true == 'gt_danger_NFF'
    assert specs.perception == specs.perception_list[0]
    assert specs.kappa == [3, 4, 5]
    assert specs.true_priori == True

    # with input
    specs2 = ral.specs_gazebo_sim(2)
    specs = ral.pt_pk_345(specs2)

    # check specs basic ok
    assert specs.horizon == 14
    # check if danger was set correctly
    assert specs.danger_true == 'gt_danger_NFF'
    assert specs.perception == specs.perception_list[0]
    assert specs.size_team == 2
    assert specs.kappa == [3, 4, 5]
    assert specs.true_priori == True


def test_pt_pu_335():
    # no input
    specs = ral.pt_pu_335()

    # check specs basic ok
    assert specs.horizon == 14
    # check if danger was set correctly
    assert specs.danger_true == 'gt_danger_NFF'
    assert specs.perception == specs.perception_list[0]
    assert specs.kappa == [3, 3, 5]
    assert specs.true_priori == False

    # with input
    specs2 = ral.specs_gazebo_sim(2)
    specs = ral.pt_pu_335(specs2)

    # check specs basic ok
    assert specs.horizon == 14
    # check if danger was set correctly
    assert specs.danger_true == 'gt_danger_NFF'
    assert specs.perception == specs.perception_list[0]
    assert specs.size_team == 2
    assert specs.kappa == [3, 3, 5]
    assert specs.true_priori == False


def test_pt_pu_333():
    # no input
    specs = ral.pt_pu_333()

    # check specs basic ok
    assert specs.horizon == 14
    # check if danger was set correctly
    assert specs.danger_true == 'gt_danger_NFF'
    assert specs.perception == specs.perception_list[0]
    assert specs.kappa == [3, 3, 3]
    assert specs.true_priori == False

    # with input
    specs2 = ral.specs_gazebo_sim(2)
    specs = ral.pt_pu_333(specs2)

    # check specs basic ok
    assert specs.horizon == 14
    # check if danger was set correctly
    assert specs.danger_true == 'gt_danger_NFF'
    assert specs.perception == specs.perception_list[0]
    assert specs.size_team == 2
    assert specs.kappa == [3, 3, 3]
    assert specs.true_priori == False



def test_pb_pu_345():

    # no input
    specs = ral.pb_pu_345()

    # check specs basic ok
    assert specs.horizon == 14
    # check if danger was set correctly
    assert specs.danger_true == 'gt_danger_NFF'
    assert specs.perception == specs.perception_list[1]
    assert specs.kappa == [3, 4, 5]
    assert specs.true_priori == False

    # with input
    specs2 = ral.specs_gazebo_sim(2)
    specs = ral.pb_pu_345(specs2)

    # check specs basic ok
    assert specs.horizon == 14
    # check if danger was set correctly
    assert specs.danger_true == 'gt_danger_NFF'
    assert specs.perception == specs.perception_list[1]
    assert specs.size_team == 2
    assert specs.kappa == [3, 4, 5]
    assert specs.true_priori == False


def test_pb_pu_335():

    # no input
    specs = ral.pb_pu_335()

    # check specs basic ok
    assert specs.horizon == 14
    # check if danger was set correctly
    assert specs.danger_true == 'gt_danger_NFF'
    assert specs.perception == specs.perception_list[1]
    assert specs.kappa == [3, 3, 5]
    assert specs.true_priori is False

    # with input
    specs2 = ral.specs_gazebo_sim(2)
    specs = ral.pb_pu_335(specs2)

    # check specs basic ok
    assert specs.horizon == 14
    # check if danger was set correctly
    assert specs.danger_true == 'gt_danger_NFF'
    assert specs.perception == specs.perception_list[1]
    assert specs.size_team == 2
    assert specs.kappa == [3, 3, 5]
    assert specs.true_priori is False


def test_pb_pu_333():

    # no input
    specs = ral.pb_pu_333()

    # check specs basic ok
    assert specs.horizon == 14
    # check if danger was set correctly
    assert specs.danger_true == 'gt_danger_NFF'
    assert specs.perception == specs.perception_list[1]
    assert specs.kappa == [3, 3, 3]
    assert specs.true_priori is False

    # with input
    specs2 = ral.specs_gazebo_sim(2)
    specs = ral.pb_pu_333(specs2)

    # check specs basic ok
    assert specs.horizon == 14
    # check if danger was set correctly
    assert specs.danger_true == 'gt_danger_NFF'
    assert specs.perception == specs.perception_list[1]
    assert specs.size_team == 2
    assert specs.kappa == [3, 3, 3]
    assert specs.true_priori is False


