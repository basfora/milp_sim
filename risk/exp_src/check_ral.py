from milp_sim.risk.exp_src import ral_default as ral
from milp_sim.risk.classes.danger import MyDanger


def print_danger_info():

    print('NFF env, fire descriptors\n------------------')

    # option for z computation, ground truth and estimation
    op = 1

    print('Human Ground Truth')
    ground_truth = 'gt_danger_NFF'
    MyDanger.print_danger_levels(ground_truth, op)

    print('Estimation, 5 % Images')
    est = 'estimate_danger_fire_des_NFF_freq_100'
    MyDanger.print_danger_levels(est, op)


if __name__ == "__main__":
    print_danger_info()
