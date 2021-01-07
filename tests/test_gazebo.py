from milp_sim.risk.classes.gazebo import MyGazeboSim
from milp_sim.risk.src import base_fun as bf
import os


def test_no_danger():
    # dummy parameters to test
    n = 46
    v_maybe = [8, 10, 12, 14, 17, 15]
    b_dummy = [0.0 for i in range(n + 1)]
    for vertex in v_maybe:
        b_dummy[vertex] = 1.0 / 6
    # current positions - all alive
    pos_dummy = [2, 3, 4]
    visited_dummy = [[1], [1], [1]]
    simulation_op = 4
    t = 0

    ms = MyGazeboSim(pos_dummy, b_dummy, visited_dummy, t, simulation_op)

    # original parameters
    assert ms.v0 == [1]
    assert ms.kappa_original == [3, 4, 5]
    assert ms.alpha_original == [0.6, 0.4, 0.4]
    assert ms.S_original == [1, 2, 3]
    assert ms.m_original == 3
    assert ms.id_map == [(1, 1), (2, 2), (3, 3)]

    assert ms.specs.danger_constraints is False
    assert ms.specs.danger_kill is False

    # input parameters
    assert ms.input_pos == [2, 3, 4]
    assert ms.bt_1 == b_dummy
    assert ms.time_step == t
    assert ms.sim_op == simulation_op

    # modified (but all searchers were alive prior to interaction
    assert ms.visited == [1, 2, 3, 4]
    assert ms.m == 3
    assert ms.kappa == [3, 4, 5]
    assert ms.alpha == [0.6, 0.4, 0.4]
    assert ms.current_pos == [2, 3, 4]

    # since it's t = 0, no kill
    assert ms.alive == [1, 2, 3]
    assert ms.killed == []
    assert ms.abort is False

    my_plan, my_belief_vector, my_visited = ms.output_results()

    assert my_visited == [1, 2, 3, 4]
    assert len(my_belief_vector) == 47

    # no effect on capture (too far)
    assert b_dummy == my_belief_vector

    del ms

    # ------------
    # next time step
    next_v = bf.next_position(my_plan)
    visited = bf.smart_list_add(my_visited, next_v)
    t = 1
    # searcher 2 got stuck
    pos_dummy = [next_v[0], -2, next_v[2]]

    # time step 1
    ms = MyGazeboSim(pos_dummy, my_belief_vector, visited, t, simulation_op)

    # original parameters (same as before)
    assert ms.v0 == [1]
    assert ms.kappa_original == [3, 4, 5]
    assert ms.alpha_original == [0.6, 0.4, 0.4]
    assert ms.S_original == [1, 2, 3]
    assert ms.m_original == 3

    # result of input parameters
    assert ms.id_map == [(1, 1), (2, -1), (3, 2)]
    assert ms.input_pos == pos_dummy
    assert ms.bt_1 == my_belief_vector
    assert ms.time_step == 1
    assert ms.sim_op == simulation_op

    # modified (2/3 searchers were alive prior to interaction)
    assert ms.visited == visited
    assert ms.current_pos == [next_v[0], next_v[2]]
    assert ms.m == 2
    assert ms.kappa == [3, 5]
    assert ms.alpha == [0.6, 0.4]

    assert ms.alive == [1, 3]
    assert ms.killed == [2]
    assert ms.abort is False

    my_plan, my_belief_vector, my_visited = ms.output_results()

    assert my_plan[2] == [-1 for i in range(14 + 1)]
    assert len(my_belief_vector) == 47

    no_change = True
    for el in ms.current_pos:
        if el in v_maybe:
            no_change = False

    # no effect on capture (too far)
    if no_change:
        assert b_dummy == my_belief_vector

    del ms


def test_change_belief():
    # dummy parameters to test
    n = 46
    v_maybe = [8, 10, 12, 14, 17, 15]
    b_dummy = [0.0 for i in range(n + 1)]
    for vertex in v_maybe:
        b_dummy[vertex] = 1.0 / 6
    visited_dummy = [[1], [1], [1]]
    simulation_op = 4
    t = 1

    # searcher 2 got stuck
    pos_dummy = [8, -2, 5]

    # time step 1
    ms = MyGazeboSim(pos_dummy, b_dummy, visited_dummy, t, simulation_op)

    # original parameters (same as before)
    assert ms.v0 == [1]
    assert ms.kappa_original == [3, 4, 5]
    assert ms.alpha_original == [0.6, 0.4, 0.4]
    assert ms.S_original == [1, 2, 3]
    assert ms.m_original == 3

    # result of input parameters
    assert ms.id_map == [(1, 1), (2, -1), (3, 2)]
    assert ms.input_pos == pos_dummy
    assert ms.bt_1 == b_dummy
    assert ms.time_step == 1
    assert ms.sim_op == simulation_op

    # modified (2/3 searchers were alive prior to interaction)
    assert ms.visited_t_1 == visited_dummy
    assert ms.current_pos == [8, 5]
    assert ms.m == 2
    assert ms.kappa == [3, 5]
    assert ms.alpha == [0.6, 0.4]

    assert ms.alive == [1, 3]
    assert ms.killed == [2]
    assert ms.abort is False

    my_plan, my_belief_vector, my_visited = ms.output_results()

    assert my_plan[2] == [-1 for i in range(14 + 1)]
    assert len(my_belief_vector) == 47

    assert my_visited == [1, 8, 5]

    assert my_belief_vector[0] == b_dummy[8]
    assert my_belief_vector[8] == 0


def test_log_file():
    # dummy parameters to test
    n = 46
    v_maybe = [8, 10, 12, 14, 17, 15]
    b_dummy = [0.0 for i in range(n + 1)]
    for vertex in v_maybe:
        b_dummy[vertex] = 1.0 / 6
    visited_dummy = [[1], [1], [1]]
    simulation_op = 1
    t = 1

    # searcher 2 got stuck
    pos_dummy = [8, -2, 5]

    log_path = os.path.dirname(os.path.abspath(__file__))

    # time step 1
    ms = MyGazeboSim(pos_dummy, b_dummy, visited_dummy, t, simulation_op, log_path)

    # original parameters (same as before)
    assert ms.v0 == [1]
    assert ms.kappa_original == [3, 4, 5]
    assert ms.alpha_original == [0.6, 0.4, 0.4]
    assert ms.S_original == [1, 2, 3]
    assert ms.m_original == 3

    # result of input parameters
    assert ms.id_map == [(1, 1), (2, -1), (3, 2)]
    assert ms.input_pos == pos_dummy
    assert ms.bt_1 == b_dummy
    assert ms.time_step == 1
    assert ms.sim_op == simulation_op

    # modified (2/3 searchers were alive prior to interaction)
    assert ms.visited_t_1 == visited_dummy
    assert ms.current_pos == [8, 5]
    assert ms.m == 2
    assert ms.kappa == [3, 5]
    assert ms.alpha == [0.6, 0.4]

    assert ms.alive == [1, 3]
    assert ms.killed == [2]
    assert ms.abort is False

    my_plan, my_belief_vector, my_visited = ms.output_results()

    assert my_plan[2] == [-1 for i in range(14 + 1)]
    assert len(my_belief_vector) == 47

    assert my_visited == [1, 8, 5]

    assert my_belief_vector[0] == b_dummy[8]
    assert my_belief_vector[8] == 0


def test_pu_pb_335():
    # dummy parameters to test
    n = 46
    v_maybe = [8, 10, 12, 14, 17, 15]
    b_dummy = [0.0 for i in range(n + 1)]
    for vertex in v_maybe:
        b_dummy[vertex] = 1.0 / 6
    # current positions - all alive
    pos_dummy = [2, 3, 4]
    visited_dummy = [[1], [1], [1]]
    simulation_op = 9
    t = 0

    ms = MyGazeboSim(pos_dummy, b_dummy, visited_dummy, t, simulation_op)

    # original parameters
    assert ms.v0 == [1]
    assert ms.kappa_original == [3, 3, 5]
    assert ms.alpha_original == [0.6, 0.6, 0.4]
    assert ms.S_original == [1, 2, 3]
    assert ms.m_original == 3
    assert ms.id_map == [(1, 1), (2, 2), (3, 3)]

    assert ms.specs.danger_constraints is True
    assert ms.specs.danger_kill is True
    # check specs basic ok
    assert ms.specs.horizon == 14
    # check if danger was set correctly
    assert ms.specs.danger_true == 'gt_danger_NFF'
    assert ms.specs.perception == ms.specs.perception_list[1]

    # input parameters
    assert ms.input_pos == [2, 3, 4]
    assert ms.bt_1 == b_dummy
    assert ms.time_step == t
    assert ms.sim_op == simulation_op

    # modified (but all searchers were alive prior to interaction
    assert ms.visited == [1, 2, 3, 4]
    assert ms.m == 3
    assert ms.kappa == [3, 3, 5]
    assert ms.alpha == [0.6, 0.6, 0.4]
    assert ms.current_pos == [2, 3, 4]

    # since it's t = 0, no kill
    assert ms.alive == [1, 2, 3]
    assert ms.killed == []
    assert ms.abort is False

    my_plan, my_belief_vector, my_visited = ms.output_results()

    assert my_visited == [1, 2, 3, 4]
    assert len(my_belief_vector) == 47

    # no effect on capture (too far)
    assert b_dummy == my_belief_vector

    del ms

    # ------------
    # next time step
    next_v = bf.next_position(my_plan)
    visited = bf.smart_list_add(my_visited, next_v)
    t = 1
    # searcher 2 got stuck
    pos_dummy = [next_v[0], -2, next_v[2]]

    # time step 1
    ms = MyGazeboSim(pos_dummy, my_belief_vector, visited, t, simulation_op)

    assert ms.specs.danger_constraints is True
    assert ms.specs.danger_kill is True
    # check specs basic ok
    assert ms.specs.horizon == 14
    # check if danger was set correctly
    assert ms.specs.danger_true == 'gt_danger_NFF'
    assert ms.specs.perception == ms.specs.perception_list[1]

    # original parameters (same as before)
    assert ms.v0 == [1]
    assert ms.kappa_original == [3, 3, 5]
    assert ms.alpha_original == [0.6, 0.6, 0.4]
    assert ms.S_original == [1, 2, 3]
    assert ms.m_original == 3

    # result of input parameters
    assert ms.id_map == [(1, 1), (2, -1), (3, 2)]
    assert ms.input_pos == pos_dummy
    assert ms.bt_1 == my_belief_vector
    assert ms.time_step == 1
    assert ms.sim_op == simulation_op

    # modified (2/3 searchers were alive prior to interaction)
    assert ms.visited == visited
    assert ms.current_pos == [next_v[0], next_v[2]]
    assert ms.m == 2
    assert ms.kappa == [3, 5]
    assert ms.alpha == [0.6, 0.4]

    assert ms.alive == [1, 3]
    assert ms.killed == [2]
    assert ms.abort is False

    my_plan, my_belief_vector, my_visited = ms.output_results()

    assert my_plan[2] == [-1 for i in range(14 + 1)]
    assert len(my_belief_vector) == 47

    no_change = True
    for el in ms.current_pos:
        if el in v_maybe:
            no_change = False

    # no effect on capture (too far)
    if no_change:
        assert b_dummy == my_belief_vector

