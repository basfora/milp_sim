from milp_sim.risk.classes.child_mespp import MyInputs2, MySearcher2, MySolverData2


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
    specs.set_theta(1)
    # solver timeout (in sec)
    specs.set_timeout(10)

    # danger stuff
    specs.set_threshold([3, 4], 'kappa')
    eta_true = [1, 3, 3, 4, 5, 3, 4, 4, 1]
    eta_priori = eta_true

    specs.set_danger_data(eta_true, eta_priori)

    return specs


def get_specs3():
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
    specs.set_theta(1)
    # solver timeout (in sec)
    specs.set_timeout(10)

    # danger stuff
    specs.set_threshold([3, 4], 'kappa')
    specs.set_threshold([0.95, 0.90], 'alpha')

    eta_true = [1, 3, 3, 4, 5, 3, 4, 4, 1]
    eta_priori = eta_true

    specs.set_danger_data(eta_true, eta_priori)
    specs.set_danger_perception('prob')

    return specs


def test_myinputs2():

    specs = get_specs()

    assert len(specs.graph.vs) == 9
    assert specs.b0 == [0, 0, 0, 0, 0, 0, 0.5, 0, 0.5, 0]
    assert specs.start_searcher_random is False
    assert specs.start_searcher_v == [1, 1]
    assert specs.horizon == 3
    assert specs.kappa == [3, 4]
    assert specs.danger_true == [1, 3, 3, 4, 5, 3, 4, 4, 1]
    assert specs.danger_priori == [1, 3, 3, 4, 5, 3, 4, 4, 1]
    assert specs.perception == 'point'


def test_specs_prob():

    specs = get_specs3()

    assert specs.alpha == [0.95, 0.90]
    assert specs.kappa == [3, 4]
    assert specs.perception == 'prob'

