"""probabilistic approach"""
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

    """[1] Perfect a priori knowledge"""
    specs = icra.specs_true_priori_prob()
    icra.num_sim(specs)
    # ---
    del specs
    i, tic = get_time(tic, i)
    
    """[2] 5% images, probabilistic"""
    specs = icra.specs_prob()
    icra.num_sim(specs)
    # ---
    del specs
    i, tic = get_time(tic, i)
    
    """[3] 100% images, probabilistic"""
    specs = icra.specs_100_img_prob()
    icra.num_sim(specs)
    # ---
    del specs
    i, tic = get_time(tic, i)

    # ---------------------
    # re run
    # ---------------------

    """[4] 100% images, point"""
    specs = icra.specs_100_img()
    icra.num_sim(specs)
    # ---
    del specs
    i, tic = get_time(tic, i)

    """[5] 5% images, point"""
    specs = icra.specs_danger_common()
    icra.num_sim(specs)
    # ---
    del specs
    i, tic = get_time(tic, i)
    
    """[6] No FOV, probabilistic"""
    specs = icra.specs_no_fov_prob()
    icra.num_sim(specs)
    # ---
    del specs
    i, tic = get_time(tic, i)

    """[7] Different team makeup"""
    # Team make up 3-3-5
    specs = icra.specs_335_prob()
    icra.num_sim(specs)
    # ---
    del specs
    i, tic = get_time(tic, i)









