"""call this for revision RAL plots"""
from milp_sim.risk.classes.risk_plot import RiskPlot


# floor plan, graph and danger - fig 5
def plot_graph_danger():
    risk_data = RiskPlot()

    ground_truth = 'gt_danger_NFF'
    estimation = 'estimate_danger_fire_des_NFF_freq_05'

    list_danger = [ground_truth, estimation]
    risk_data.set_danger_files(list_danger)
    risk_data.print_danger_data()
    risk_data.plot_graph_school()


def compile_configs_op1():
    """250 instances per config
    outcomes, times, losses in a row
    """

    """Input name of folder in data_saved and instance base"""
    n_runs = 250
    rp = RiskPlot(n_runs)

    folder_names = ['0106-NC-NK-NA', '0106-PT-PK-345',
                    '0106-PB-PK-345', '0106-PT-PU-345',
                    '0106-PB-PU-345', '0106-NC-DK-NA']

    instance_base = ['NCNKHT_MGTdesFF_NApoint_G46Vss2_0106_', 'DCDKHT_MGTdesFF_100hpoint_PK_G46Vss2_0106_',
                     'DKDCPB_PK345_desFF_NA_G46Vss2_0106_', 'DCDKHT_MGTdesFF_05hpoint_G46Vss2_0105_',
                     'DKDCPB_PU345_desFF_05_G46Vss2_0106_', 'NCDKHT_MGTdesFF_NAhpoint_G46Vss2_0106_']

    for i in range(len(folder_names)):
        rp.retrieve_data(folder_names[i], instance_base[i])


def plot_configs_op1():

    """250 instances per config
    outcomes, times, losses in a row
    """
    """Input name of pickle file in data_compiled"""
    n_runs = 250
    rp = RiskPlot(n_runs)

    pickle_list = ['0106-NC-NK-NA', '0106-PT-PK-345',
                   '0106-PB-PK-345', '0106-PT-PU-345',
                   '0106-PB-PU-345', '0106-NC-DK-NA']

    # rp.retrieve_outcomes(pickle_list)
    rp.retrieve_casualties(pickle_list)
    rp.retrieve_times(pickle_list)


def compile_configs_op2():
    """250 instances per config
    outcomes, times, losses in a row
    different team makeups
    """

    """Input name of folder in data_saved and instance base"""
    n_runs = 250
    rp = RiskPlot(n_runs)

    folder_names = ['0106-PT-PU-335', '0106-PT-PU-333',
                    '0106-PB-PU-335', '0106-PB-PU-333']

    instance_base = ['DCDKHT_MGTdesFF_05hpoint_G46Vss2_0106_', 'DCDKHH_MGTdesFF_05hpoint_G46Vss2_0106_',
                     'DKDCPB_PU335_desFF_05_G46Vss2_0106_', 'DKDCPB_PU333_desFF_05_G46Vss2_0106_']

    for i in range(len(folder_names)):
        if i == 0:
            continue
        rp.retrieve_data(folder_names[i], instance_base[i])


def compile_1000_op2():
    """250 instances per config
    outcomes, times, losses in a row
    different team makeups
    """

    """Input name of folder in data_saved and instance base"""
    n_runs = 1000
    rp = RiskPlot(n_runs)

    folder_names = ['0107-PT-PU-335', '0108-PT-PU-333',
                    '0108-PB-PU-335', '0108-PB-PU-333']

    instance_base = ['DKDCPT_PU335_desFF_05_G46Vss2_0107_', 'DKDCPT_PU333_desFF_05_G46Vss2_0108_',
                     'DKDCPB_PU335_desFF_05_G46Vss2_0108_', 'DKDCPB_PU333_desFF_05_G46Vss2_0109_']

    for i in range(len(folder_names)):
        rp.retrieve_data(folder_names[i], instance_base[i])


def plot_configs_op2():

    """250 instances per config
    outcomes, times, losses in a row
    """
    """Input name of pickle file in data_compiled"""
    n_runs = 250
    rp = RiskPlot(n_runs)

    pickle_list = ['0106-PT-PU-345', '0106-PT-PU-335', '0106-PT-PU-333',
                   '0106-PB-PU-345', '0106-PB-PU-335', '0106-PB-PU-333']

    # rp.retrieve_outcomes(pickle_list)
    # rp.retrieve_casualties(pickle_list)
    rp.retrieve_times(pickle_list)


def plot_1000_op2():

    """250 instances per config
    outcomes, times, losses in a row
    """
    """Input name of pickle file in data_compiled"""
    n_runs = 1000
    rp = RiskPlot(n_runs)

    pickle_list = ['0107-PT-PU-345', '0107-PT-PU-335', '0108-PT-PU-333',
                   '0107-PB-PU-345', '0108-PB-PU-335', '0108-PB-PU-333']

    # rp.retrieve_outcomes(pickle_list)
    # rp.retrieve_casualties(pickle_list)
    rp.retrieve_times(pickle_list)


def plot_1000_op1():
    """250 instances per config
     outcomes, times, losses in a row
     """
    """Input name of pickle file in data_compiled"""
    n_runs = 1000
    rp = RiskPlot(n_runs)

    pickle_list = ['0107-NC-NK-NA', '0107-PT-PK-345',
                   '0107-PB-PK-345', '0107-PT-PU-345',
                   '0107-PB-PU-345', '0107-NC-DK-NA']

    # rp.retrieve_outcomes(pickle_list)
    # rp.retrieve_casualties(pickle_list)
    rp.retrieve_times(pickle_list)


def compile_1000_op1():
    """250 instances per config
    outcomes, times, losses in a row
    """

    """Input name of folder in data_saved and instance base"""
    n_runs = 1000
    rp = RiskPlot(n_runs)

    folder_names = ['0107-NC-NK-NA', '0107-PT-PK-345',
                    '0107-PB-PK-345', '0107-PT-PU-345',
                    '0107-PB-PU-345', '0107-NC-DK-NA']

    instance_base = ['NKNC_desFF_NA_G46Vss2_0107_', 'DKDCPT_PK345_desFF_NA_G46Vss2_0107_',
                     'DKDCPB_PK345_desFF_NA_G46Vss2_0108_', 'DKDCPT_PU345_desFF_05_G46Vss2_0106_',
                     'DKDCPB_PU345_desFF_05_G46Vss2_0108_', 'DKNC_desFF_NA_G46Vss2_0107_']

    for i in range(len(folder_names)):
        rp.retrieve_data(folder_names[i], instance_base[i])


if __name__ == "__main__":
    # plot_graph_danger()
    # compile_configs_op1()
    # compile_configs_op2()
    # plot_configs_op1()
    # plot_configs_op2()
    # compile_1000_op1()
    # plot_1000_op1()
    # compile_1000_op2()
    plot_1000_op2()
