from milp_sim.risk.classes.danger import MyDanger
from milp_sim.risk.classes.child_mespp import MyInputs2


def test_compute_H():

    eta = [0.1, 0.3, 0.4, 0.7, 0.9]

    a = MyDanger.sum_1(eta)
    assert a is False

    eta = [0.1, 0.3, 0.2, 0.2, 0.2]
    b = MyDanger.sum_1(eta)
    assert b is True

    list_k = [2, 4]

    # sum for each level
    H2 = MyDanger.compute_H(eta)
    assert H2 == [0.1, 0.4, 0.6, 0.8, 1.0]

    H1 = MyDanger.compute_H(eta, list_k)
    my_sum2 = 0.1 + 0.3
    my_sum4 = 0.1 + 0.3 + 0.2 + 0.2
    assert H1 == [my_sum2, my_sum4]


def test_argmax_eta():

    eta_v = [0.1, 0.3, 0.3, 0.2, 0.1]
    z_list = MyDanger.argmax_eta(eta_v)
    assert MyDanger.sum_1(eta_v) is True
    assert z_list == [2, 3]

    eta_v = [0.1, 0.2, 0.3, 0.2, 0.2]
    z_list = MyDanger.argmax_eta(eta_v)
    assert MyDanger.sum_1(eta_v) is True
    assert z_list == [3]


def test_z_from_eta():

    eta1 = [0.1, 0.3, 0.3, 0.2, 0.1]
    eta2 = [0.3, 0.3, 0.3, 0.05, 0.05]

    b = MyDanger.sum_1(eta1)
    c = MyDanger.sum_1(eta2)
    assert b is True
    assert c is True

    # mean
    op1 = 1
    z1 = MyDanger.z_from_eta(eta1, op1)
    assert z1 == 3
    z1 = MyDanger.z_from_eta(eta2, op1)
    assert z1 == 2

    # max
    op2 = 2
    z2 = MyDanger.z_from_eta(eta1, op2)
    assert z2 == 3
    z2 = MyDanger.z_from_eta(eta2, op2)
    assert z2 == 3

    # min kappa
    op3 = 3
    kappa1 = [2]
    kappa2 = [3]
    z3 = MyDanger.z_from_eta(eta1, op3, kappa1)
    z4 = MyDanger.z_from_eta(eta1, op3, kappa2)
    assert z3 == 2
    assert z4 == 3

    eta_1 = MyDanger.eta_from_z(z1)
    assert eta_1 == [0.1, 0.6, 0.1, 0.1, 0.1]

    eta = [0.2, 0.2, 0.2, 0.2, 0.2]
    k = 3
    z = MyDanger.z_from_eta(eta, 4, k)
    assert z == 4


def test_eta_from_z():
    list_z = [1, 1, 2, 3, 4, 5]

    eta = []

    for z in list_z:
        eta_v = MyDanger.eta_from_z(z, 1.0)
        eta.append(eta_v)

    assert eta[0] == [1, 0, 0, 0, 0]
    assert eta[1] == [1, 0, 0, 0, 0]
    assert eta[2] == [0, 1, 0, 0, 0]
    assert eta[3] == [0, 0, 1, 0, 0]
    assert eta[4] == [0, 0, 0, 1, 0]
    assert eta[5] == [0, 0, 0, 0, 1]


def test_frequentist():

    img1 = [0.1, 0.1, 0.6, 0.1, 0.1]
    img2 = [0.1, 0.1, 0.6, 0.1, 0.1]
    img3 = [0.1, 0.1, 0.6, 0.1, 0.1]

    assert MyDanger.sum_1(img1) is True
    assert MyDanger.sum_1(img2) is True
    assert MyDanger.sum_1(img3) is True

    xi = dict()
    xi[1] = [img1, img2, img3]

    eta_hat, z_hat = MyDanger.compute_frequentist(xi)

    assert z_hat[0] == 3
     
    assert eta_hat[0] == [0.1, 0.1, 0.6, 0.1, 0.1]


def test_compute_apriori():

    n = 4
    op_avg = 1
    op_conservative = 2
    op_min_k = 3
    op_mva = 4

    # default with weighted avg
    my_eta1 = None
    eta0_0, z0_0 = MyDanger.compute_from_value(n, my_eta1, op_avg)
    assert z0_0 == [3, 3, 3, 3]
    for v_idx in range(n):
        assert z0_0[v_idx] == 3
        assert eta0_0[v_idx] == [0.2, 0.2, 0.2, 0.2, 0.2]

    k = 3
    eta0_0, z0_0 = MyDanger.compute_from_value(n, my_eta1, op_mva, k)
    for v_idx in range(n):
        assert z0_0[v_idx] == 4
        assert eta0_0[v_idx] == [0.2, 0.2, 0.2, 0.2, 0.2]

    # one danger for all vertices
    my_eta2 = 2
    eta0_0, z0_0 = MyDanger.compute_from_value(n, my_eta2, op_avg)
    assert z0_0 == [2, 2, 2, 2]
    for v_idx in range(n):
        assert z0_0[v_idx] == 2
        assert eta0_0[v_idx] == [0.1, 0.6, 0.1, 0.1, 0.1]

    # one danger level for each vertex
    my_eta3 = [1, 2, 3, 5]
    eta0_0, z0_0 = MyDanger.compute_from_value(n, my_eta3, op_avg)
    assert z0_0 == [1, 2, 3, 5]
    assert eta0_0[0] == [0.6, 0.1, 0.1, 0.1, 0.1]
    assert eta0_0[1] == [0.1, 0.6, 0.1, 0.1, 0.1]
    assert eta0_0[2] == [0.1, 0.1, 0.6, 0.1, 0.1]
    assert eta0_0[3] == [0.1, 0.1, 0.1, 0.1, 0.6]

    # prob for each vertex
    my_eta4 = [[0.1, 0.2, 0.3, 0.3, 0.1], [0.3, 0.1, 0.3, 0.1, 0.2], [0.1, 0.2, 0.3, 0.3, 0.1], [0.1, 0.2, 0.3, 0.3, 0.1]]
    eta0_0, z0_0 = MyDanger.compute_from_value(n, my_eta4, op_conservative)
    assert z0_0 == [4, 3, 4, 4]
    for v_idx in range(n):
        assert eta0_0[v_idx] == my_eta4[v_idx]

    # break ties - z == min k value
    z = MyDanger.z_from_eta(eta0_0[0], op_min_k, [3, 4, 5])
    assert z == 3


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


def test_mva():
    print('hello')

def test_create_and_estimate():

    specs = get_specs()

    eta_true = [1, 3, 3, 4, 5, 3, 4, 4, 1]
    eta_priori = [2, 2, 2, 3, 3, 3, 2, 2, 2]

    specs.set_danger_data(eta_true, 'true')
    specs.set_danger_data(eta_priori, 'priori')

    g = specs.graph

    danger_true = specs.danger_true
    danger_priori = specs.danger_priori

    assert danger_true == [1, 3, 3, 4, 5, 3, 4, 4, 1]
    assert danger_priori == [2, 2, 2, 3, 3, 3, 2, 2, 2]

    # will set lookup equal to a priori
    eta_hat_2 = 0

    # create class
    danger = MyDanger(g)
    # set danger values (ground truth, estimate and a priori)
    danger.set_true(danger_true)
    danger.set_priori(danger_priori)
    danger.set_estimate(eta_hat_2)

    assert danger.n == len(g.vs)
    assert danger.z0_0 == danger_priori
    assert danger.z == danger_true
    assert danger.perception == 'point'
    assert danger.z_hat == danger_priori
    assert danger.lookup_z_hat == danger_priori

    del danger

    # ----------------------------------------------------------
    # will set estimate lookup equal to true value
    eta_hat_1 = None

    # create class
    danger = MyDanger(g)
    # set danger values (ground truth, estimate and a priori)
    danger.set_true(danger_true)
    danger.set_estimate(eta_hat_1)
    danger.set_priori(danger_priori)

    assert danger.n == len(g.vs)
    assert danger.z0_0 == danger_priori
    assert danger.z == danger_true
    assert danger.perception == 'point'
    assert danger.z_hat == danger_priori
    assert danger.lookup_z_hat == danger_true

    for v in range(1, danger.n + 1):
        v_idx = v - 1
        assert danger.get_z(v) == danger_true[v_idx]
        assert danger.get_zhat(v) == danger_priori[v_idx]
        # lookup table: equal to true value (eta_hat_1 = None)
        eta_hat, z_hat, H_hat = danger.get_from_lookup(v)
        assert z_hat == danger_true[v_idx]

    # test estimate function
    op_test = True
    danger.set_use_fov(True, op_test)

    visited_vertices = [1]
    danger.estimate(visited_vertices)
    danger_estimate = [1, 1, 2, 1, 3, 3, 2, 2, 2]
    for v in range(1, 10):
        assert danger.get_z(v) == danger_true[v - 1]
        assert danger.get_zhat(v) == danger_estimate[v - 1]

    visited_vertices = [1, 2]
    danger.estimate(visited_vertices)
    danger_estimate = [1, 3, 3, 1, 3, 3, 2, 2, 2]
    for v in range(1, 10):
        assert danger.get_z(v) == danger_true[v - 1]
        assert danger.get_zhat(v) == danger_estimate[v - 1]

    visited_vertices = [1, 3]
    danger.estimate(visited_vertices)
    danger_estimate = [1, 1, 3, 1, 3, 3, 2, 2, 2]

    for v in range(1, 10):
        assert danger.get_z(v) == danger_true[v - 1]
        assert danger.get_zhat(v) == danger_estimate[v - 1]

















