import risk.src.base_fun
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


if __name__ == "__main__":
    # default
    # n = 4
    # my_eta1 = None
    # eta0_0, z0_0 = MyDanger.compute_apriori(n, my_eta1)
    # specs = get_specs()
    # path = plnr.run_planner(specs)
    #fov = risk.src.base_fun.fov_ss2()
    #print(fov)

    #print(fov[8])
    eta_v = [0.1, 0.3, 0.3, 0.2, 0.2]
    z_list = MyDanger.argmax_eta(eta_v)
    print(z_list)

