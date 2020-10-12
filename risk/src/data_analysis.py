from milp_sim.risk.classes.risk_plot import RiskPlot
from milp_sim.risk.classes.stats import MyStats
from milp_sim.risk.classes.danger import MyDanger
from milp_sim.risk.src import base_fun as bf
import copy

"""Code for analysing data from exp batch exp_1009, consisting of:
exp_15: NC DK worst case scenario, big prob
exp_16: DC DK HT (should protect MVP), big prob
exp_12: NC DK worst case scenario, small prob"""


def analysis_exp_1010():

    # code for simulation folders
    datename = ['1009', '1010']
    probname = ['b', 's']
    pername = [5, 100]
    config = ['NCDKHT', 'DCDKHT']
    n_runs = 1000

    rp = RiskPlot()
    parent_folders = list()
    parent_folders.append(rp.assemble_parent_name(datename[0], config[0], probname[0]))
    parent_folders.append(rp.assemble_parent_name(datename[1], config[1], probname[0]))

    parent_folders.append(rp.assemble_parent_name(datename[1], config[0], probname[1]))
    parent_folders.append(rp.assemble_parent_name(datename[1], config[1], probname[1]))

    # NC-b
    stats_nc_b = MyStats(parent_folders[0], n_runs, pername[1])
    # MyStats(parent_folders[0], n_runs, pername[0])

    # DC-b
    stats_dc_b = MyStats(parent_folders[1], n_runs, pername[1])
    # MyStats(parent_folders[1], n_runs, pername[0])

    # rp.add_stats(stats_nc_b)
    # rp.add_stats(stats_dc_b)
    rp.plot_stat([stats_nc_b.cum_success_rate, stats_dc_b.cum_success_rate], 'Success Rate', ['NC', 'DC'])

    # # NC-s
    # stats_nc_s = MyStats(parent_folders[2], n_runs, pername[1])
    # MyStats(parent_folders[2], n_runs, pername[0])
    #
    # # DC-s
    # stats_dc_s = MyStats(parent_folders[3], n_runs, pername[1])
    #
    # rp.add_stats(stats_nc_s)
    # rp.add_stats(stats_dc_s)
    # rp.plot_stat()

    return

def plot_only():

    extension = '.pkl'
    folder_path = MyDanger.get_folder_path('data_compiled')
    low_prob = ['1010NCDKHT-s-100', '1010DCDKHT-s-100']
    high_prob = ['1009NCDKHT-b-100', '1009NCDKHT-b-05', '1010DCDKHT-b-100', '1010DCDKHT-b-05']
    f_names = low_prob
    success_rate = []
    casualty_mission_rate = []
    avg_mission_time = []
    casualty_mvp = []

    for f_name in f_names:
        f_path = folder_path + '/' + f_name + extension
        stats = bf.load_pickle_file(f_path)
        # ---
        stats.print_name(stats.parent_name, '--\nStats for')
        stats.print_final_stats()

        success_rate.append(copy.deepcopy(stats.cum_success_rate))
        casualty_mission_rate.append(copy.deepcopy(stats.cum_casualty_mission_rate))
        avg_mission_time.append(copy.deepcopy(stats.cum_mission_time))
        casualty_mvp.append(copy.deepcopy(stats.cum_casualty_mvp_rate))

        print(stats.prob_kill)

        del stats

    rp = RiskPlot()

    for success in success_rate:
        rp.add_stats(success)

    rp.plot_stat()
    rp.clear_list()
    #
    for avg_time in avg_mission_time:
        rp.add_stats(avg_time)
    #
    rp.plot_stat()
    rp.clear_list()

    for casual in casualty_mission_rate:
        rp.add_stats(casual)

    rp.plot_stat()
    rp.clear_list()

    for mvp in casualty_mvp:
        rp.add_stats(mvp)

    rp.plot_stat()
    rp.clear_list()


if __name__ == '__main__':
    analysis_exp_1010()
    # plot_only()

