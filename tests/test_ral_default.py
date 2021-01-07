"""Test if ral_default is setting the right things"""

from milp_sim.risk.exp_src import ral_default as ral


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


def test_num_sim():

    specs = ral.specs_num_sim()

    assert specs.horizon == 14
    assert specs.size_team == 3
    assert specs.start_searcher_v == [1, 1, 1]


def test_default_thresholds():
    kappa, alpha = ral.default_thresholds()

    assert kappa == [3, 4, 5]
    assert alpha == [0.6, 0.4, 0.4]


def test_common_danger():
    specs = ral.specs_danger_common()

    assert specs.kappa == [3, 4, 5]
    assert specs.alpha == [0.6, 0.4, 0.4]
    assert specs.danger_kill is True
    assert specs.danger_constraints is True


def test_no_danger():

    specs = ral.specs_no_danger()

    assert specs.danger_kill is False
    assert specs.danger_constraints is False

    # default thresholds are still stored there
    assert specs.kappa == [3, 4, 5]
    assert specs.alpha == [0.6, 0.4, 0.4]


