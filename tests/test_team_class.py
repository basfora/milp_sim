from milp_sim.risk.classes.danger import MyDanger
from milp_sim.risk.classes.child_mespp import MyInputs2
from milp_sim.risk.src import risk_parameters as rp
from milp_sim.risk.src import plan_risk as plnr


def get_specs():

    specs = MyInputs2()
    specs.set_graph(4)

    # solver parameter: central x distributed
    specs.set_solver_type('distributed')
    # target motion
    specs.set_target_motion('static')
    # searchers' detection: capture range and false negatives
    m = 2
    specs.set_capture_range(0)
    specs.set_size_team(m)
    # position
    v0 = [1, 1]
    specs.set_start_searchers(v0)
    b_0 = [0.0 for i in range(10)]
    b_0[8] = 0.5
    b_0[6] = 0.5
    specs.set_b0(b_0)

    # time-step stuff: deadline mission (tau), planning horizon (h), re-plan frequency (theta)
    h = 3

    specs.set_all_times(h)
    # specs.set_theta(3)
    # solver timeout (in sec)
    specs.set_timeout(10)

    # danger stuff
    specs.set_threshold([3, 4], 'kappa')
    eta_true = [1, 3, 3, 4, 5, 3, 4, 4, 1]
    eta_priori = eta_true

    specs.set_danger_data(eta_true, eta_priori)

    return specs


def test_create_team():
    specs = get_specs()

    team = rp.create_searchers(specs)

    assert team.S == [1, 2]
    assert team.m == 2
    for s_id in team.S:
        s = team.searchers[s_id]
        assert s.capture_range == 0
        assert s.zeta is None
        assert s.start == 1
        assert s.current_pos == 1

        if s_id == 1:
            assert s.kappa == 3
        if s_id == 2:
            assert s.kappa == 4

    assert team.current_positions == {1: 1, 2: 1}
    assert team.start_positions == [1, 1]

    assert team.kappa == [3, 4]
    assert team.kappa_original == [3, 4]

    s1 = team.searchers[1]
    s1.set_alive(False)

    assert team.searchers[1].alive is False
    assert team.searchers_original[1].alive is True

    s1 = team.searchers[1]
    s1.set_new_id(3)

    assert team.searchers[1].id == 3
    assert team.searchers_original[1].id == 1
    assert team.searchers[1].id_0 == 1









