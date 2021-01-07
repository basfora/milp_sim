from milp_sim.risk.exp_src import ral_default as ral
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

    """NC DK no danger scenario"""
    # no constraints (should be worse)
    specs = ral.specs_no_danger()
    ral.num_sim(specs)
    # ---
    del specs
    i, tic = get_time(tic, i)











