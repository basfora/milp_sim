from milp_sim.risk.src import plan_risk as plnr, base_fun as bf
from milp_sim.risk.classes.child_mespp import MyInputs2
from milp_sim.risk.src import risk_parameters as rp
from milp_mespp.core import extract_info as ext

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

    specs.set_danger_data(eta_true, 'true')
    specs.set_danger_data(eta_priori, 'priori')

    return specs


def get_specs2():

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
    b_0[9] = 1.0
    specs.set_b0(b_0)

    # time-step stuff: deadline mission (tau), planning horizon (h), re-plan frequency (theta)
    h = 4

    specs.set_all_times(h)
    specs.set_theta(1)
    # solver timeout (in sec)
    specs.set_timeout(10)

    # danger stuff
    specs.set_threshold([3, 4], 'kappa')
    eta_check = [1, 3, 3, 4, 5, 3, 4, 4, 1]
    eta_priori = eta_check

    specs.set_danger_data(eta_priori, 'true')
    specs.set_danger_data(eta_check, 'priori')


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
    specs.set_threshold([0.95, 0.95], 'alpha')

    eta_true = [1, 3, 3, 4, 5, 3, 4, 4, 1]
    eta_priori = eta_true

    specs.set_danger_data(eta_true, 'true')
    specs.set_danger_data(eta_priori, 'priori')
    specs.set_danger_perception(1)

    return specs


def test_rparam():
    specs = get_specs()
    team = rp.create_searchers(specs)
    searchers = team.searchers

    assert len(searchers.keys()) == 2

    for s_id in searchers.keys():
        s = searchers[s_id]
        assert s.kappa == specs.kappa[s_id - 1]

    kappa_list = rp.get_kappa(searchers)
    assert kappa_list == specs.kappa


def test_run_planner():

    specs = get_specs()

    path_list = plnr.run_planner(specs)

    assert path_list[1] == [1, 2, 3, 6]
    assert path_list[2] == [1, 4, 7, 8]

    specs = get_specs2()

    path_list = plnr.run_planner(specs)

    assert path_list[1] == [1, 2, 3, 6, 9]
    assert path_list[2][0] == 1
    assert path_list[2][1] == 4
    assert path_list[2][2] == 7


def test_init_wrapper():
    specs = get_specs()

    belief, team, solver_data, target, danger = plnr.init_wrapper(specs)

    assert team.m_input == 2
    assert team.S == [1, 2]
    assert list(team.searchers.keys()) == [1, 2]


def test_fov():

    rooms = bf.compartments_ss2()
    my_list = []
    for name in rooms.keys():
        for v in rooms[name]:
            my_list.append(v)

    my_list.sort()
    V = ext.get_set_vertices(46)[0]

    assert my_list == V





