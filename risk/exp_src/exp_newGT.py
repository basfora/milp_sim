"""probabilistic approach"""
from milp_sim.risk.exp_src import icra_default as icra
import time


def get_timer():
    return time.perf_counter()


def get_time(tic, i=1):
    min_ = 60
    toc = get_timer()
    delta_t = round((toc - tic) / min_, 2)

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

    """[1] New ground truth I5"""
    specs = icra.specs_new_true_point()
    icra.num_sim(specs)
    # ---
    del specs
    i, tic = get_time(tic, i)

    """[1] New ground truth I5 335"""
    specs = icra.specs_new_true_335()
    icra.num_sim(specs)
    # ---
    del specs
    i, tic = get_time(tic, i)











