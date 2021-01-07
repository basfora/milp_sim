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
    """1009NCDKHT-b"""

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


def analysis_exp_1011():
    # TODO fix naming! - parent
    # name format 1009NCDKHT-b
    rp = RiskPlot()

    datename = ['1012']
    config = ['DCDKHT']
    probname = ['h']
    pername = [100]
    n_runs = 485
    add = ['PK']

    rp.retrieve_data(datename, config, probname, pername, n_runs, add)

    return None


def analysis_exp_1012():

    rp = RiskPlot()

    datename = ['1012', '1013']
    config = ['DCDKHT', 'NCDKHT']
    extra_id = ['PK', '100', '05', 'NA', 'NFOV']

    pername = [5, 100]
    n_runs = 1000
    perception = 'point'

    folders = []

    """1012DCDKHT-h-PK"""
    folders.append(rp.assemble_parent_name(datename[0], config[0], extra_id[0])[0])
    """1012DCDKHT-h-100"""
    folders.append(rp.assemble_parent_name(datename[0], config[0], extra_id[1])[0])
    """1012DCDKHT-h-05"""
    folders.append(rp.assemble_parent_name(datename[0], config[0], extra_id[2])[0])
    """1012NCDKHT-h-NA"""
    folders.append(rp.assemble_parent_name(datename[0], config[1], extra_id[3])[0])

    folders.append('1013NCNKHT-n-0')

    """1013NCDKHT-h-NFOV """
    folders.append(rp.assemble_parent_name(datename[1], config[0], extra_id[4])[0])
    #
    subtitle = ['Perfect Knowledge', '100\% images', '5\% images', 'No Constraints', 'No Danger', 'no FOV']

    # rp.retrieve_data(folders[0], 'DCDKHT_100hpoint_PK_G46Vss2_1012_', subtitle[0])
    # rp.retrieve_data(folders[1], 'DCDKHT_100hpoint_G46Vss2_1012_', subtitle[1])
    # rp.retrieve_data(folders[2], 'DCDKHT_05hpoint_G46Vss2_1012_', subtitle[2])
    # rp.retrieve_data(folders[3], 'NCDKHT_NAhpoint_G46Vss2_1012_', subtitle[3])
    # rp.retrieve_data(folders[4], 'NCNKHT_NApoint_G46Vss2_1013_', subtitle[4])

    subtitle.append('no FOV')
    rp.retrieve_data(folders[5], 'DCDKHT_05hpoint_NFOV_G46Vss2_1013_', subtitle[4])


def plot_for_paper():
    # 3 plots mission outcome, casualties, mission times
    rp = RiskPlot()
    #  [ND, PK, PU, NC]
    #  [0, 1, 2, 3, 4]

    # fig 1
    pickle_list = ['1111DCDKHT_HGTFC-PT-05', '1111DCDKHT_HGTFF-PT-05', '1111DCDKHT_MGTFF-PT-05', '1111DCDKHT_MGTFC-PT-05']
    # ['1013NCNKHT-n-0', '1012DCDKHT-h-PK',  '1015DCDKHT-PK-prob', '1012DCDKHT-h-05', '1015DCDKHHT-05-PT', '1015DCDKHT-05-prob', '1012NCDKHT-h-NA']

    # FIG 2
    # pickle_list = ['1012DCDKHT-h-05', ]

    rp.retrieve_outcomes(pickle_list)
    rp.retrieve_casualties(pickle_list)
    rp.retrieve_times(pickle_list)


def compile_data():
    """Input name of folder in saved_data and instance base"""
    # folder_names = ['1015DCDKHHT-05-PT'] # ['1015DCDKHT-PK-prob', '1015DCDKHT-05-prob']
    # instance_base = ['DCDKHT_05hpoint_G46Vss2_1015_']  #  ['DCDKHT_100hprob_PK_G46Vss2_1015_', 'DCDKHT_05hprob_G46Vss2_1015_']

    rp = RiskPlot()

    folder_names = ['0106-PT-PU-345', '1111DCDKHT_MGTFC-PT-05', '1111DCDKHT_HGTFF-PT-05',
                    '1111DCDKHT_MGTFF-PT-05']
    instance_base = ['DCDKHT_HGTdesFC_05hpoint_G46Vss2_1110_', 'DCDKHT_MGTdesFC_05hpoint_G46Vss2_1110_',
                     'DCDKHT_HGTdesFF_05hpoint_G46Vss2_1111_', 'DCDKHT_MGTdesFF_05hpoint_G46Vss2_1111_']

    for i in range(len(folder_names)):
        rp.retrieve_data(folder_names[i], instance_base[i])


if __name__ == '__main__':

    # plot_for_paper()
    compile_data()



