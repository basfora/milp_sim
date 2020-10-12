from milp_sim.risk.exp_src import icra_default as icra
import time


def get_timer():
    return time.perf_counter()


def get_time(tic, i=1):
    min_ = 60
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

    """DC DK PK"""
    # perfect priori
    specs = icra.specs_true_priori()
    icra.num_sim(specs)
    # ---
    del specs
    i, tic = get_time(tic, i)

    """NC DK worst case scenario"""
    # no constraints (should be worse)
    specs = icra.specs_no_constraints()
    icra.num_sim(specs)
    # ---
    del specs
    i, tic = get_time(tic, i)

    """DC DK 5 HT"""
    # normal estimate (5%)
    specs = icra.specs_danger_common()
    icra.num_sim(specs)
    # ---
    del specs
    i, tic = get_time(tic, i)

    """DC DK NOFOV"""
    # no fov (just to test)
    specs = icra.specs_no_fov()
    icra.num_sim(specs)
    # ---
    del specs
    i, tic = get_time(tic, i)






