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

    specs.set_danger_data(eta_true, 'true')
    specs.set_danger_data(eta_priori, 'priori')

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


def test_searchers_update():
    specs = get_specs()

    team = rp.create_searchers(specs)

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


def test_get_paths():
    specs = get_specs()
    sim_data = True
    belief, target, team, solver_data, danger, inf = plnr.run_planner(specs, sim_data)

    path_list = team.get_path_list()
    path_dict = team.get_path()

    assert path_list[1] == [1, 2, 3, 6]
    assert path_list[2] == [1, 4, 7, 8]

    assert path_dict[(1, 0)] == 1
    assert path_dict[(1, 1)] == 2
    assert path_dict[(1, 2)] == 3
    assert path_dict[(1, 3)] == 6

    assert path_dict[(2, 0)] == 1
    assert path_dict[(2, 1)] == 4
    assert path_dict[(2, 2)] == 7
    assert path_dict[(2, 3)] == 8


def test_decide_searchers_luck():
    specs = get_specs()
    v0 = [2, 3]
    eta_true = [1, 3, 3, 4, 5, 3, 4, 4, 1]
    specs.set_start_searchers(v0)

    belief, team, solver_data, target, danger = plnr.init_wrapper(specs)

    v_1, v_2 = 2, 3
    v_1_idx, v_2_idx = 1, 2
    assert team.searchers[1].current_pos == v_1
    assert team.searchers[2].current_pos == v_2

    assert danger.z == eta_true

    assert danger.get_z(v_1) == eta_true[v_1_idx] == 3
    assert danger.get_z(v_2) == eta_true[v_2_idx] == 3

    prob_list = [0.1, 0.2, 1, 1, 1]
    danger.set_prob_kill(prob_list)

    assert danger.is_fatal(v_1) is True
    assert danger.is_fatal(v_2) is True

    assert team.alive == [1, 2]

    # start the killing spree
    t = 0
    killed_ids = team.to_kill_or_not_to_kill(danger, t)

    # when all are killed -- related vars
    assert killed_ids == [1, 2]
    assert team.killed == [1, 2]
    assert team.number_casualties == 2
    assert team.killed_info[1] == [v_1, t, 3, 3]
    assert team.killed_info[2] == [v_2, t, 3, 4]
    assert list(team.searchers_killed.keys()) == [1, 2]

    # alive vars (not yet updated)
    assert len(team.alive) == 2

    # update dict
    team.update_searchers_ids()
    assert len(team.searchers) == 0

    # update team size
    team.update_size(len(team.searchers))
    assert team.m == 0
    assert team.S == []

    # update alive list
    team.update_alive()
    assert len(team.alive) == 0

    # positions (prior to update)
    assert team.current_positions == {1: 2, 2: 3}
    # update positions
    team.update_pos_list()
    assert team.current_positions == {}

    # thresholds (prior to update)
    assert team.kappa == [3, 4] == team.kappa_original
    assert team.alpha == [0.95, 0.95] == team.alpha_original

    # update thresholds
    team.update_kappa()
    assert team.kappa == []

    # update alpha
    team.update_alpha()
    assert team.alpha == []


def test_decide_searchers_luck2():
    specs = get_specs()
    v0 = [4, 3]
    z_true = [4, 3]
    eta_true = [1, 3, 3, 4, 5, 3, 4, 4, 1]
    specs.set_start_searchers(v0)

    belief, team, solver_data, target, danger = plnr.init_wrapper(specs)

    v_1, v_2 = v0[0], v0[1]
    v_1_idx, v_2_idx = v_1 - 1, v_2 - 1
    assert team.searchers[1].current_pos == v_1
    assert team.searchers[2].current_pos == v_2

    assert danger.z == eta_true

    assert danger.get_z(v_1) == eta_true[v_1_idx] == 4
    assert danger.get_z(v_2) == eta_true[v_2_idx] == 3

    prob_list = [0.1, 0.2, 0, 1, 1]
    danger.set_prob_kill(prob_list)

    # kill s_1, keep s_2
    assert danger.is_fatal(v_1) is True
    assert danger.is_fatal(v_2) is False

    assert team.alive == [1, 2]

    # start the killing spree
    t = 0
    killed_ids = team.to_kill_or_not_to_kill(danger, t)

    # when all are killed -- related vars
    assert killed_ids == [1]
    assert team.killed == [1]
    assert team.number_casualties == 1
    assert team.killed_info[1] == [v_1, t, 4, 3]
    assert list(team.searchers_killed.keys()) == [1]

    # alive vars (not yet updated)
    assert len(team.alive) == 2

    # update dict
    team.update_searchers_ids()
    assert len(team.searchers) == 1
    assert team.searchers[1].id == 1
    assert team.searchers[1].id_0 == 2

    # update team size
    team.update_size(len(team.searchers))
    assert team.m == 1
    assert team.S == [1]

    # update alive list
    team.update_alive()
    assert len(team.alive) == 1

    # positions (prior to update)
    assert team.current_positions == {1: 4, 2: 3}
    # update positions
    team.update_pos_list()
    assert team.current_positions == {1: 3}

    # thresholds (prior to update)
    assert team.kappa == [3, 4] == team.kappa_original
    assert team.alpha == [0.95, 0.95] == team.alpha_original

    # update thresholds
    team.update_kappa()
    assert team.kappa == [4]

    # update alpha
    team.update_alpha()
    assert team.alpha == [0.95]













