"""probabilistic approach"""
from milp_sim.risk.exp_src import icra_default as icra
import time


def get_timer():
    return time.perf_counter()


def get_time(tic, i=1):
    min_ = 3600
    toc = get_timer()
    delta_t = round((toc - tic)/min_, 2)

    print("----------------------------------------------------------------------------------------------------")
    print("----------------------------------------------------------------------------------------------------")
    print('Batch %d done in %.2f min' % (i, delta_t))
    print("----------------------------------------------------------------------------------------------------")
    print("----------------------------------------------------------------------------------------------------\n")

    i += 1
    tic = get_timer()
    return i, tic


if __name__ == "__main__":

    i = 1
    tic = get_timer()

    """Perfect a priori knowledge"""
    specs = icra.specs_true_priori_prob()
    icra.num_sim(specs)
    # ---
    del specs
    i, tic = get_time(tic, i)

    """5% images, probabilistic"""
    specs = icra.specs_prob()
    icra.num_sim(specs)
    # ---
    del specs
    i, tic = get_time(tic, i)

    """100% images, probabilistic"""
    specs = icra.specs_100_img_prob()
    icra.num_sim(specs)
    # ---
    del specs
    i, tic = get_time(tic, i)

    """No FOV, probabilistic"""
    specs = icra.specs_no_fov_prob()
    icra.num_sim(specs)
    # ---
    del specs
    i, tic = get_time(tic, i)

    """Different team makeup"""
    # Team make up 3-3-5
    specs = icra.specs_335_prob()
    icra.num_sim(specs)
    # ---
    del specs
    i, tic = get_time(tic, i)









