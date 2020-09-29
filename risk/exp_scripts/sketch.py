from milp_sim.risk.classes.danger import MyDanger



if __name__ == "__main__":
    # default
    n = 4
    my_eta1 = None
    eta0_0, z0_0 = MyDanger.compute_apriori(n, my_eta1)
