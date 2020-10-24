from milp_sim.risk.classes.gazebo import MyGazeboSim
from milp_sim.risk.src import base_fun as bf


def test_common_point():
    # dummy parameters to test
    n = 46
    v_maybe = [8, 10, 12, 14, 17, 15]
    b_dummy = [0.0 for i in range(n + 1)]
    for vertex in v_maybe:
        b_dummy[vertex] = 1.0 / 6
    # current positions - all alive
    pos_dummy = [2, 3, 4]
    visited_dummy = [1]
    simulation_op = 4
    t = 0

    ms = MyGazeboSim(pos_dummy, b_dummy, visited_dummy, t, simulation_op)

    # original parameters
    assert ms.v0 == [1]
    assert ms.kappa_original == [3, 4, 5]
    assert ms.alpha_original == [0.95, 0.95, 0.95]
    assert ms.S_original == [1, 2, 3]
    assert ms.m_original == 3
    assert ms.id_map == [(1, 1), (2, 2), (3, 3)]

    assert ms.specs.danger_constraints is False
    assert ms.specs.danger_kill is False

    # input parameters
    assert ms.input_pos == [2, 3, 4]
    assert ms.b_0 == b_dummy
    assert ms.time_step == t
    assert ms.sim_op == simulation_op

    # modified (but all searchers were alive prior to interaction
    assert ms.visited == [1, 2, 3, 4]
    assert ms.m == 3
    assert ms.kappa == [3, 4, 5]
    assert ms.alpha == [0.95, 0.95, 0.95]
    assert ms.current_pos == [2, 3, 4]

    # since it's t = 0, no kill
    assert ms.alive == [1, 2, 3]
    assert ms.killed == []
    assert ms.abort is False

    my_plan, my_belief_vector, my_visited = ms.output_results()

    assert my_visited == [1, 2, 3, 4]
    assert len(my_belief_vector) == 47

    # next time step
    next_v = bf.next_position(my_plan)
    visited = bf.smart_list_add(my_visited, next_v)
    t = 1
    pos_dummy = [next_v[0], -1, next_v[2]]

    del ms

    ms = MyGazeboSim(pos_dummy, my_belief_vector, visited, t, simulation_op)

    # original parameters
    assert ms.v0 == [1]
    assert ms.kappa_original == [3, 4, 5]
    assert ms.alpha_original == [0.95, 0.95, 0.95]
    assert ms.S_original == [1, 2, 3]
    assert ms.m_original == 3

    # input parameters
    assert ms.id_map == [(1, 1), (2, -1), (3, 2)]
    assert ms.input_pos == pos_dummy
    assert ms.b_0 == my_belief_vector
    assert ms.time_step == 1
    assert ms.sim_op == simulation_op

    # modified (but all searchers were alive prior to interaction
    assert ms.visited == visited
    assert ms.current_pos == [next_v[0], next_v[2]]
    assert ms.m == 2
    assert ms.kappa == [3, 5]
    assert ms.alpha == [0.95, 0.95]

    assert ms.alive == [1, 3]
    assert ms.killed == [2]
    assert ms.abort is False

    my_plan, my_belief_vector, my_visited = ms.output_results()




















