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
    L = [1, 2, 3, 4, 5]

    H1 = MyDanger.compute_H(eta, list_k)
    H2 = MyDanger.compute_H(eta)

    assert H1 == [0.4, 0.8]
    assert H2 == [0.1, 0.4, 0.6, 0.8, 1.0]


def test_z_eta_convert():
    eta = [0.1, 0.3, 0.2, 0.2, 0.2]
    b = MyDanger.sum_1(eta)
    assert b is True

    z1 = MyDanger.z_from_eta(eta)
    assert z1 == 2

    eta1 = MyDanger.eta_from_z(z1)
    assert eta1 == [0.1, 0.6, 0.1, 0.1, 0.1]


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

    # default
    my_eta1 = None
    eta0_0, z0_0 = MyDanger.compute_apriori(n, my_eta1)
    assert z0_0 == [3, 3, 3, 3]
    for v in range(n):
        assert z0_0[v] == 3
        assert eta0_0[v] == [0.2, 0.2, 0.2, 0.2, 0.2]

    # one danger for all vertices
    my_eta2 = 2
    eta0_0, z0_0 = MyDanger.compute_apriori(n, my_eta2)
    assert z0_0 == [2, 2, 2, 2]
    for v in range(n):
        assert z0_0[v] == 2
        assert eta0_0[v] == [0.1, 0.6, 0.1, 0.1, 0.1]

    # one danger level for each vertex
    my_eta3 = [1, 2, 3, 5]
    eta0_0, z0_0 = MyDanger.compute_apriori(n, my_eta3)
    assert z0_0 == [1, 2, 3, 5]
    assert eta0_0[0] == [0.6, 0.1, 0.1, 0.1, 0.1]
    assert eta0_0[1] == [0.1, 0.6, 0.1, 0.1, 0.1]
    assert eta0_0[2] == [0.1, 0.1, 0.6, 0.1, 0.1]
    assert eta0_0[3] == [0.1, 0.1, 0.1, 0.1, 0.6]

    # prob for each vertex
    my_eta4 = [[0.1, 0.2, 0.3, 0.3, 0.1], [0.3, 0.1, 0.3, 0.1, 0.2], [0.1, 0.2, 0.3, 0.3, 0.1], [0.1, 0.2, 0.3, 0.3, 0.1]]
    eta0_0, z0_0 = MyDanger.compute_apriori(n, my_eta4)
    assert z0_0 == [4, 3, 4, 4]
    for v in range(n):
        assert eta0_0[v] == my_eta4[v]

    # break ties - z == min k value
    z = MyDanger.z_from_eta(eta0_0[0], [3, 4, 5], 2)
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

    specs.set_danger_data(eta_true, eta_priori)

    return specs


def test_create_danger():

    specs = get_specs()
    g = specs.graph
    danger_true = specs.danger_true
    danger_priori = specs.danger_priori

    assert danger_true == [1, 3, 3, 4, 5, 3, 4, 4, 1]
    assert danger_priori == [1, 3, 3, 4, 5, 3, 4, 4, 1]

    danger = MyDanger(g, danger_true, danger_priori)

    assert danger.n == len(g.vs)
    assert danger.z0_0 == danger_priori
    assert danger.z == danger_true
    assert danger.perception == 'point'

    assert danger.z_hat == danger_priori














