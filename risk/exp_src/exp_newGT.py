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

    """[1] NFF both description, human GT 
    New ground truth PU-PT-HGT"""
    specs = icra.specs_NFF_both()
    icra.num_sim(specs)
    # ---
    del specs
    i, tic = get_time(tic, i)

    """[2] NFF both description, model GT (100% images)
    New ground truth PU-PT-MGT"""
    specs = icra.specs_NFF_both_est()
    icra.num_sim(specs)
    # ---
    del specs
    i, tic = get_time(tic, i)

    """[3] NFF fire description only, human GT  
    New ground truth PU-PT-HGT"""
    specs = icra.specs_NFF_fire()
    icra.num_sim(specs)
    # ---
    del specs
    i, tic = get_time(tic, i)

    """[2] NFF both description only, model GT (100% images)
       New ground truth PU-PT-MGT"""
    specs = icra.specs_NFF_fire_est()
    icra.num_sim(specs)
    # ---
    del specs
    i, tic = get_time(tic, i)











