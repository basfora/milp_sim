import sys
import os
import matplotlib.pyplot as plt
import pickle
import numpy as np
from scipy.stats import sem, t
from scipy import mean as mean_sp
from milp_mespp.core import extract_info as ext
# from core import retrieve_data as rd


this_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(this_path)


# --------------------------------------------------------------------------------------------------------------------
# basic plot functions
# this is a generic function to draw a box plot.
def draw_box_plot(data, ax, edge_color, fill_color, offset=None):

    if offset is not None:
        pos = np.array(range(len(data)))*2.0+offset
        bp = ax.boxplot(data, patch_artist=True, positions=pos, widths=0.6)

    else:
        # actual plotting
        bp = ax.boxplot(data, patch_artist=True)

    for element in ['boxes', 'whiskers', 'fliers', 'means', 'medians', 'caps']:
        plt.setp(bp[element], color=edge_color)
    # plt.setp(bp['medians'], linewidth=2)

    for patch in bp['boxes']:
        patch.set(facecolor=fill_color)


def draw_errorbar(xvalues, yvalues, y_err, ax, l_style='bo-', lgd=None):
    ax.errorbar(xvalues, yvalues, yerr=y_err, fmt=l_style, label=lgd, capsize=2)
    ax.grid(b=True, which='both')


def plot_and_close(fig, filename, semi=False):

    if semi:
        plt.rcParams.update({'ytick.minor.size': 0})
        plt.rcParams.update({'ytick.minor.width': 0})
        plt.semilogy()

    else:
        plt.plot()
    plt.close()
    # plt.show()
    # Save the figure
    fig.savefig(filename, bbox_inches='tight')


# set plot colors etc
def plot_gaps(gaps, x_values, param, offset=None):

    # unpack
    x_label = param['labels'][0]
    y_label = param['labels'][1]
    filename = param['filename']
    f_size = param['f_size']

    fig = plt.figure()
    ax = fig.add_subplot(111)

    my_color = 'red'

    # box plot with edge = black, fill = red
    draw_box_plot(gaps, ax, 'black', my_color, offset)

    # plt.show()

    # thicks
    ax.set_xticklabels(map(lambda x: str(x), x_values))
    # ax.set_yticks(range(0, 30, 5))
    # ax.yaxis.set_major_locator(plt.MaxNLocator(5))

    # labels
    ax.set_x_ticks(x_label, fontsize=f_size)

    if offset is None:
        # regular tick and label
        ax.set_ylabel(y_label, fontsize=f_size)
    elif offset == 2:
        ax.set_ylim(-0.5, 32)
        # no tick, no label
        ax.set_yticklabels([])
        ax.set_yticks([])
    else:
        ax.set_ylim(-0.5, 32)
        # no label, regular tick
        pass

    plot_and_close(fig, filename)


def plot_times(times, x_values, param, offset=None):

    # unpack
    x_label = param['labels'][0]
    y_label = param['labels'][1]
    filename = param['filename']
    f_size = param['f_size']

    fig = plt.figure()
    ax = fig.add_subplot(111)

    my_color = 'blue'

    # box plot with edge = black, fill = blue
    draw_box_plot(times, ax, 'black', my_color)

    # plt.show()

    avg, std_devs = get_avg(times)

    print('avg : ' + str(avg))
    print('std : ' + str(std_devs))

    # ticks
    # ax.set_yticks(range(0, 2000, 200))
    ax.set_xticklabels(map(lambda x: str(x), x_values))

    # plot labelsax.set_ylabel(y_label, fontsize=15)
    ax.set_x_ticks(x_label, fontsize=f_size)

    if offset is None:
        # regular tick and label
        ax.set_ylabel(y_label, fontsize=f_size)
    elif offset == 2:
        # no tick, no label
        ax.set_yticklabels([])
        ax.set_yticks([])
    else:
        # no label, regular tick
        pass



    # if offset is not None:
    #     if offset < 0:
    #         plt.plot()
    #     else:
    #         plot_and_close(fig, filename)
    # else:
    plot_and_close(fig, filename)


def plot_line(objs, x_values, param):

    # unpack
    x_label = param['labels'][0]
    y_label = param['labels'][1]
    filename = param['filename']
    f_size = param['f_size']

    # avg, std_devs = get_avg(objs)
    # confidence interval
    avg, std_devs = get_confidence(objs)

    short_list = [round(avg[i], 4) for i in range(len(avg))]
    short_list_std = [round(std_devs[i], 4) for i in range(len(std_devs))]
    print('avg: ' + str(short_list))
    print('std: ' + str(short_list_std))

    fig = plt.figure()

    ax = fig.add_subplot(111)
    draw_errorbar(x_values, avg, std_devs, ax)

    # thicks
    ax.set_xticks(x_values)
    ax.set_ylim(0, max(avg) + max(std_devs)+0.1)

    # plot labels
    ax.set_x_ticks(x_label, fontsize=f_size)
    ax.set_ylabel(y_label, fontsize=f_size)

    plot_and_close(fig, filename)


def plot_multi_line(objs, x_values, param):
    """:param objs: dictionary with 2 x 5 lists each """

    # unpack
    x_label = param['labels'][0]
    y_label = param['labels'][1]
    filename = param['filename']
    lgd = param['lgd']
    l_style = param['l_style']
    lgd_pos = param['lgd_pos']
    f_size = param['f_size']

    avg = {}
    std = {}
    y_err = {}
    y_avg = {}

    fig = plt.figure()

    ax = fig.add_subplot(111)

    for key in objs.keys():

        # regular average
        avg[key], std[key] = get_avg(objs[key])

        # confidence interval
        y_avg[key], y_err[key] = get_confidence(objs[key])

        d_avg = get_difference1D(avg[key], y_avg[key])

        if max(d_avg) > 0.02:
            # catch for error in calculation
            print('avg1 --> ' + str(avg[key]))
            print('avg2 --> ' + str(y_avg[key]))
        else:
            short_list = [round(y_avg[key][i], 4) for i in range(len(y_avg[key]))]
            short_list_std = [round(y_err[key][i], 4) for i in range(len(y_err[key]))]
            print('avg -> ' + str(short_list))
            print('std -> ' + str(short_list_std))

        # PLOT
        draw_errorbar(x_values, y_avg[key], y_err[key], ax, l_style[key], lgd[key])

    # thicks
    ax.set_xticks(x_values)
    # ax.set_ylim(-1, 1)

    # plot labels
    ax.set_x_ticks(x_label, fontsize=f_size)
    ax.set_ylabel(y_label, fontsize=f_size)

    # lgd
    plt.legend(loc=lgd_pos)

    semi = param['log']

    plot_and_close(fig, filename, semi)


# --------------------------------------------------------------------------------------------------------------------
# handling info

def get_difference(list1, list2):

    my_size = len(list1)
    my_range = list(range(0, my_size))
    diff = []

    if len(list2) != my_size:
        print('Error! lists have different sizes')
        exit()
    else:
        for i in my_range:

            diff.append([])

            A = list1[i]
            B = list2[i]

            my_2_size = list(range(0, len(A)))

            for j in my_2_size:
                diff[i].append(A[j] - B[j])

    return diff


def get_difference1D(list1, list2):

    my_size = len(list1)
    my_range = list(range(0, my_size))
    diff = []

    if len(list2) != my_size:
        print('Error! lists have different sizes')
        exit()
    else:
        for i in my_range:

            A = list1[i]
            B = list2[i]

            diff.append(A-B)

    return diff


def get_confidence(some_list: list):
    """Get confidence interval of 95%"""

    confidence = 0.68

    m = []
    y_err = []
    max_j = len(some_list)

    for i in range(0, max_j):
        n = len(some_list[i])
        m_i = mean_sp(some_list[i][:])
        std_err = sem(some_list[i][:])
        h = std_err * t.ppf((1 + confidence) / 2, n - 1)

        m.append(m_i)
        y_err.append(h)

    return m, y_err


def get_avg(some_list: list):

    avg = []
    std = []
    max_j = len(some_list)

    for i in range(0, max_j):
        avg.append(np.mean(some_list[i][:]))
        std.append(np.std(some_list[i][:]))

    return avg, std


def get_median(some_list: list):
    mdn = []
    iqr = []
    max_j = len(some_list)

    for i in range(0, max_j):
        x = some_list[i][:]

        x_m = np.median(x)
        mdn.append(x_m)

        q75, q25 = np.percentile(x, [75, 25])
        my_iqr = q75 - q25
        iqr.append(my_iqr)

    return mdn, iqr


def get_per_difference(list1, list2):
    """list1 : quocient (base value)"""
    my_size = len(list1)
    my_range = list(range(0, my_size))
    diff = []

    if len(list2) != my_size:
        print('Error! lists have different sizes')
        exit()
    else:
        for i in my_range:

            diff.append([])

            A = list1[i]
            B = list2[i]

            my_2_size = list(range(0, len(A)))

            for j in my_2_size:
                quo = A[j]
                per_ = abs(100*(A[j] - B[j]))/quo
                diff[i].append(per_)
    return diff


# retrieves the results (pickle file) of a particular experiment.
def get_actual_data(base, n_run, log_path, subfolders):
    """
    :param base: the sub folder name right before the _DATE_InstanceNumber
    :param n_run: the INSTANCE number in the subfolder name
    :param log_path: path to the main log folder containing all the runs of an experiment (e.g. ../data/CH6-14S1G1TNSV/)
    :param subfolders: the list of all the sub folders contained in log_folder
    :param log_path: complete path
    :return:
    """
    for subfolder in subfolders:
        splitted = subfolder.split('_')
        # get basename, compare to base; compare n_run with experiment instance
        if splitted[0] == base and str(n_run).zfill(3) == splitted[2]:
            filepath = log_path + '/' + subfolder + '/global_save.txt'
            try:
                data = pickle.load(open(filepath, "rb"))
            except:
                print('Make sure your parameters are right!')
                data = None
                exit()

            return data


def get_from_data(data, t: int):

    try:
        if data['solver_data'].solver_type != 'central':
            sol_time = ext.get_last_info(data['solver_data'].solve_time[t])[1]
            sol_gap = ext.get_last_info(data['solver_data'].gap[t])[1]
            sol_obj = ext.get_last_info(data['solver_data'].obj_value[t])[1]
        else:
            sol_time = ext.get_last_info(data['solver_data'].solve_time)[1]
            sol_gap = ext.get_last_info(data['solver_data'].gap)[1]
            sol_obj = ext.get_last_info(data['solver_data'].obj_value)[1]
    except:
        print('Error')

    return sol_time, sol_gap, sol_obj


def get_mission_time(data, not_found=0):

    if data['target'].is_captured is False:
        not_found = 1
        mission_time = data['solver_data'].deadline
    else:
        mission_time = data['target'].capture_time

    return mission_time, not_found


def get_number_file(n_runs, j, i, key_l):

    n_id = j

    # fix the numbering if needed
    if key_l == 'H' and j == 2:
        n_id = 0
    elif key_l == 'H' and j > 2:
        n_id = j-1

    if key_l == 'H' and n_runs == 200:
        n_id = 0

    n_file = n_id * n_runs + i

    return n_file


def add_to_record_1(mip_gaps, mip_times, obj_fcn, data, t):

    # data of interest for plots
    sol_time, sol_gap, sol_obj = get_from_data(data, t)

    # add to the record
    mip_times.append(sol_time)
    mip_gaps.append(100 * sol_gap)
    obj_fcn.append(sol_obj)

    return mip_gaps, mip_times, obj_fcn


def add_to_record_2(obj, mission_t, data, t):

    # data of interest for plots
    sol_obj = get_from_data(data, t)[2]
    sol_mission, not_found = get_mission_time(data)

    # append to list
    obj.append(sol_obj)
    mission_t.append(sol_mission)

    return obj, mission_t, not_found


def count_not_found(n_not_found, data):

    if data['target'].is_captured is False:
        n_not_found += 1

    return n_not_found


def check_data():

    name_folder = ['DH5T50S1-5G2FNMV']

    log_path = '../plot_data/'

    folder_interest = {}
    subfolders = {}
    for folder in name_folder:
        # get path for the parent folder and subfolders in it
        folder_interest[folder] = log_path + folder

        # sub folders of my folder of interest
        subfolders[folder] = os.listdir(folder_interest[folder])

    complete_path = folder_interest
    subfolders = subfolders

    n_runs = 100
    list_runs = list(range(1, n_runs + 1))
    folder = name_folder[0]

    all_obj_fcn, all_mission_times = {}, {}

    all_obj_fcn[folder], all_mission_times[folder], all_not_found = [], [], []

    for j in [0, 1, 2, 3, 4]:
        # get name of file, ex CH10S1G1TNSV
        filename = 'DH5T50S' + str(j+1) + 'G2FNMV'

        # initialize
        obj_fcn, mission_times, n_not_found = [], [], 0

        # loop through instances
        for i in list_runs:

            # get number of file
            n_file = get_number_file(n_runs, j, i, 'S')

            # retrieve data from file
            data = get_actual_data(filename, n_file, complete_path[folder], subfolders[folder])

            if n_file == 1:
                print('searcher seed: ' + str(data['searchers'][1].seed) + ' target seed: ' + str(data['target'].seed))

            if data is None:
                print('Error! ' + filename)
                continue
            else:
                # we want the data from time = 0
                t = 0
                obj_fcn, mission_times, not_found = add_to_record_2(obj_fcn, mission_times, data, t)

                n_not_found += not_found

        all_not_found.append(n_not_found)
        all_mission_times[folder].append(mission_times)
        all_obj_fcn[folder].append(obj_fcn)

        # name, labels, value of figure to be saved
        # all the figures are saved in the ''figs'' folder (make sure to create it)
    print('not found: ' + str(all_not_found))

    x_values = [1, 2, 3, 4, 5]
    param = {}

    y_label = 'Avg Mission Time [steps]'
    s1 = 'all_endtime_'
    filename_save = '../figs/' + s1 + folder + '.pdf' #'.png'
    x_label = '$m$'
    labels = [x_label, y_label]

    param['labels'] = labels
    param['filename'] = filename_save

    plot_line(all_mission_times[folder], x_values, param)


# --------------------------------------------------------------------------------------------------------------------
# parent plotting

def fig_paddle(op, settings, folder=''):

    param = {}

    x_values = settings.my_list

    if op == 'obj':
        y_label = 'Reward [avg]'
        s1 = 'obj_'

    elif op == 'end_time':
        y_label = 'Mission Time [steps]'
        s1 = 'endtime_'

    elif op == 'mip_gaps':
        y_label = 'MIP Gap [ \% ]'
        s1 = 'gaps_'

    elif op == 'mip_times':
        y_label = 'Solution Time [ s ]'
        s1 = 'times_'

    elif op == 'end_diff':
        y_label = 'Mission Time Difference [steps]'
        s1 = 'delta_time_'

    elif op == 'obj_diff':
        y_label = 'Avg Reward Loss [ \% ]'
        s1 = 'delta_obj_'

    elif op == 'obj_together':
        y_label = 'Reward [avg]'
        s1 = 'all_obj_'

    elif op == 'end_together':
        y_label = 'Avg Mission Time [steps]'
        s1 = 'all_endtime_'

    elif op == 'gaps_together':
        y_label = 'MIP Gap [ \% ]'
        s1 = 'all_gaps'

    elif op == 'times_together':
        y_label = 'Avg Solution Time [ s ]'
        s1 = 'all_times'

    elif op == 'times_together_cpp':
        y_label = 'Avg Solution Time [ s ]'
        s1 = 'all_times'

    elif op == 'times_diff':
        y_label = 'Solution Time [ s ] \ndifference between h = 5, 10 [ s ]'
        s1 = 'diff_times'

    elif op == 'times_diff_cpp':
        y_label = 'Solution Time \n Decrease with MILP [ \% ]'
        s1 = 'diff_times'

    elif op == 'gaps_diff':
        y_label = 'MIP Gaps [ \% ]'
        s1 = 'diff_gaps'

    elif op == 'mip_times_fit':
        y_label = ''
        s1 = 'times_'

    elif op == 'mip_gaps_fit':
        y_label = ''
        s1 = 'gaps_'

    else:
        y_label = ''
        s1 = ''

    if 'diff' in op:
        lgd = []

        code = settings.code
        n_lines = len(code[2:])

        aux_folder = str(code[0]) + 'x' + str(code[1])
        aux_folder2 = ''

        for i in range(0, n_lines):
            aux_folder2 = aux_folder2 + '_' + code[i+2]

            lgd.append(settings.name_scenes[i])
            # l_style.append(list_styles[i])

            folder = aux_folder + aux_folder2

        param['lgd'] = lgd

    if 'together' in op:
        lgd = []

        i = -1
        for folder in settings.parent:
            i += 1

            splitted2 = folder.split('G')
            aux3 = 'G' + splitted2[-1]

            if 'cpp' in op:
                if '_cpp' in folder:
                    aux2 = ', SoA'
                    aux3 = aux3.split('_')[0]
                else:
                    aux2 = ', MILP'
            else:
                splitted = folder.split('T')
                if len(splitted[0]) == 3:
                    aux2 = ", h=5"
                elif len(splitted[0]) == 4:
                    aux2 = ", h=10"
                else:
                    splitted = folder.split('H')
                    aux1 = splitted[0]
                    aux2 = "-" + aux1 + ", h =10"

            aux1 = settings.decode_scene(aux3)
            lgd.append(aux1 + aux2)

        if not settings.lgd:
            for i in range(len(lgd)):
                lgd[i] = None

        param['lgd'] = lgd

    list_styles = settings.lines_color
    lgd_loc = settings.lgd_loc

    param['l_style'] = list_styles
    param['lgd_pos'] = lgd_loc

    filename = '../figs/' + s1 + folder + '.pdf'
    x_label = settings.x_label
    labels = [x_label, y_label]

    param['labels'] = labels
    param['filename'] = filename
    param['f_size'] = 18

    if settings.log:
        param['log'] = True
    else:
        param['log'] = False

    return x_values, param


def plot_difference(settings, all_obj_fcn, all_mission_times, op='obj'):
    code = settings.code

    n_lines = len(code[2:])

    objs = {}
    times = {}

    for i in range(0, n_lines):

        A, B, C, D = [], [], [], []
        name1 = ''
        name2 = ''

        for folder in settings.parent:
            if code[0] in folder and code[i + 2] in folder:
                A = all_obj_fcn[folder]
                C = all_mission_times[folder]
                name1 = folder
            if code[1] in folder and code[i + 2] in folder:
                B = all_obj_fcn[folder]
                D = all_mission_times[folder]
                name2 = folder

        print('....\nGetting difference between ' + name1 + ' and ' + name2)
        objs[i] = get_per_difference(A, B)
        times[i] = get_per_difference(C, D)

    if 'obj' in op:
        x_values1, param1 = fig_paddle('obj_diff', settings)
        x_values2, param2 = fig_paddle('end_diff', settings)
    else:
        x_values1, param1 = fig_paddle('gaps_diff', settings)
        x_values2, param2 = fig_paddle('times_diff', settings)

    plot_multi_line(objs, x_values1, param1)
    plot_multi_line(times, x_values2, param2)


def plot_difference_cpp(settings, all_obj_fcn, all_mission_times, op='obj'):

    code = settings.code
    n_lines = len(code[2:])

    objs = {}
    times = {}

    for i in range(0, n_lines):

        A, B, C, D = [], [], [], []
        name1 = ''
        name2 = ''

        for folder in settings.parent[i:]:

            g = folder.split('S3')[-1]

            if g in folder[i:]:
                A = all_obj_fcn[folder]
                C = all_mission_times[folder]
                name1 = folder
                break

        for folder in settings.parent[i:]:

            if g in folder and code[1] in folder:
                B = all_obj_fcn[folder]
                D = all_mission_times[folder]
                name2 = folder
                break

        print('....\nGetting difference between ' + name1 + ' and ' + name2)
        # objs[i] = get_per_difference(B, A)
        times[i] = get_per_difference(D, C)

    if 'obj' in op:
        # x_values1, param1 = fig_paddle('obj_diff', settings)
        x_values2, param2 = fig_paddle('end_diff', settings)
    else:
        # x_values1, param1 = fig_paddle('gaps_diff', settings)
        x_values2, param2 = fig_paddle('times_diff_cpp', settings)

    # plot_multi_line(objs, x_values1, param1)
    plot_multi_line(times, x_values2, param2)


def plot_together(settings, all_obj_fcn, all_mission_times, op='obj'):

    objs = {}
    times = {}

    i = -1
    for folder in settings.parent:
        i += 1

        objs[i] = all_obj_fcn[folder]
        times[i] = all_mission_times[folder]

        print('....\nExtracting info - ' + folder)

    if 'obj' in op:
        x_values1, param1 = fig_paddle('obj_together', settings)
        x_values2, param2 = fig_paddle('end_together', settings)
        value_of = ['Reward', 'Mission Time']
    else:
        if settings.cpp:
            x_values1, param1 = fig_paddle('gaps_together_cpp', settings)
            x_values2, param2 = fig_paddle('times_together_cpp', settings)
        else:
            x_values1, param1 = fig_paddle('gaps_together', settings)
            x_values2, param2 = fig_paddle('times_together', settings)
        value_of = ['Gaps', 'Sol Time']

    print('Avg %s' % value_of[0])
    plot_multi_line(objs, x_values1, param1)
    print('Avg %s' %value_of[1])
    plot_multi_line(times, x_values2, param2)


def plot_box(settings):

    """Plot box plot"""

    # collect info on each setting (e.g CH10S1-5G1TNSV)
    all_mip_gaps, all_mip_times, all_obj_fcn = {}, {}, {}

    for folder in settings.parent:

        k = settings.parent.index(folder)

        all_mip_gaps[folder],  all_mip_times[folder], all_obj_fcn[folder] = [], [], []

        print('----\n' + folder)

        # loop through variable parameter (m or h)
        for my_par in settings.my_list:
            # counter
            j = settings.my_list.index(my_par)

            # get name of file, ex CH10S1G1TNSV
            filename = settings.base[0][k] + str(my_par) + settings.base[1][k]

            # initialize
            mip_gaps, mip_times, obj_fcn, n_not_found = [], [], [], 0

            # loop through instances
            for i in settings.list_runs:

                # get number of file
                n_file = get_number_file(settings.n_runs, j, i, settings.key_l)

                # retrieve data from file
                data = get_actual_data(filename, n_file, settings.complete_path[folder], settings.subfolders[folder])

                if n_file == 1:
                    print('searcher seed: ' + str(data['searchers'][1].seed) + ' target seed: ' + str(data['target'].seed))

                # we want the data from time = 0
                t = 0
                mip_gaps, mip_times, obj_fcn = add_to_record_1(mip_gaps, mip_times, obj_fcn, data, t)

                del data

            all_mip_gaps[folder].append(mip_gaps)
            all_mip_times[folder].append(mip_times)
            all_obj_fcn[folder].append(obj_fcn)

        offset = None
        if settings.fit is True:
            if 'G2TNMV' in settings.scenario:
                if 'G2TNMV' in folder:
                    # no label
                    offset = 1
                elif 'G2FNMV' in folder:
                    # no label, no ticks
                    offset = 2
                else:
                    pass
            else:
                if 'G2FNMV' in folder:
                    # no label
                    offset = 1
                else:
                    pass

        x_values, param = fig_paddle('mip_gaps', settings, folder)
        plot_gaps(all_mip_gaps[folder], x_values, param, offset)

        x_values, param = fig_paddle('mip_times', settings, folder)
        plot_times(all_mip_times[folder], x_values, param, offset)

        print('\nMEDIAN gaps: ' + str(get_median(all_mip_gaps[folder])[0]))

        # x_values, param = fig_paddle('mip_times', settings, folder)
        # plot_times(all_mip_times[folder], x_values, param, offset)
        print('MEDIAN time: ' + str(get_median(all_mip_times[folder])[0]) + '\n')
        #
        # x_values, param = fig_paddle('obj', settings, folder)
        # plot_obj(all_obj_fcn[folder], x_values, param)


def plot_bar(settings):
    """plot delta in mission time and objective function"""

    all_obj_fcn, all_mission_times = {}, {}

    for folder in settings.parent:

        k = settings.parent.index(folder)

        all_obj_fcn[folder], all_mission_times[folder] = [], []
        all_not_found = []

        print('----\n' + folder)

        # loop through variable parameter (m or h)
        for my_par in settings.my_list:
            # counter
            j = settings.my_list.index(my_par)

            # get name of file, ex CH10S1G1TNSV
            filename = settings.base[0][k] + str(my_par) + settings.base[1][k]

            # initialize
            obj_fcn, mission_times, n_not_found = [], [], 0

            # loop through instances
            for i in settings.list_runs:
                # get number of file
                n_file = get_number_file(settings.n_runs, j, i, settings.key_l)

                # retrieve data from file
                data = get_actual_data(filename, n_file, settings.complete_path[folder], settings.subfolders[folder])

                if n_file == 1:
                    print('searcher seed: ' + str(data['searchers'][1].seed) + ' target seed: ' + str(data['target'].seed))

                # we want the data from time = 0
                t = 0
                obj_fcn, mission_times, not_found = add_to_record_2(obj_fcn, mission_times, data, t)

                n_not_found += not_found

                del data

            all_not_found.append(n_not_found)
            all_mission_times[folder].append(mission_times)
            all_obj_fcn[folder].append(obj_fcn)

        # name, labels, value of figure to be saved
        # all the figures are saved in the ''figs'' folder (make sure to create it)
        print('not found: ' + str(all_not_found))

        if settings.difference is False:
            x_values, param = fig_paddle('obj', settings, folder)
            plot_line(all_obj_fcn[folder], x_values, param)

            x_values, param = fig_paddle('end_time', settings, folder)
            plot_line(all_mission_times[folder], x_values, param)
        else:
            continue

    if settings.difference is True:
        plot_difference(settings, all_obj_fcn, all_mission_times)

    if settings.together is True:
        plot_together(settings, all_obj_fcn, all_mission_times)


def plot_bar_times(settings):
    """plot delta in mip_gaps and solving time"""

    # collect info on each setting (e.g CH10S1-5G1TNSV)
    all_mip_gaps, all_mip_times, all_obj_fcn = {}, {}, {}

    for folder in settings.parent:

        k = settings.parent.index(folder)

        if '_cpp' in folder:
            k = settings.parent.index(folder.split('_cpp')[0])

        all_mip_gaps[folder], all_mip_times[folder] = [], []

        print('----\n' + folder)

        # loop through variable parameter (m or h)
        for my_par in settings.my_list:
            # counter
            j = settings.my_list.index(my_par)

            # get name of file, ex CH10S1G1TNSV
            filename = settings.base[0][k] + str(my_par) + settings.base[1][k]

            # initialize
            mip_gaps, mip_times, obj_fcn, n_not_found = [], [], [], 0

            # loop through instances
            for i in settings.list_runs:
                # get number of file
                n_file = get_number_file(settings.n_runs, j, i, settings.key_l)

                if '_cpp' in folder:
                    date_exp = '0512'
                    if filename.split('G')[-1] == '3TNSV':
                        date_exp = '0518'
                    ins_path = settings.log_path + folder + '/' + filename + '_' + date_exp + '_' + str(n_file).zfill(3)
                    cpp_solver = rd.get_from_txt(ins_path, 'solver_data')
                    mip_times.append(cpp_solver[1][0])
                    mip_gaps.append(0.0)

                else:
                    # retrieve data from file
                    data = get_actual_data(filename, n_file, settings.complete_path[folder], settings.subfolders[folder])
                    # we want the data from time = 0
                    t = 0
                    mip_gaps, mip_times, obj_fcn = add_to_record_1(mip_gaps, mip_times, obj_fcn, data, t)

                    del data

                if n_file == 1:
                    print(filename)
                    # print('searcher seed: ' + str(data['searchers'][1].seed) + ' target seed: ' + str(data['target'].seed))

            all_mip_gaps[folder].append(mip_gaps)
            all_mip_times[folder].append(mip_times)
#            all_obj_fcn[folder].append(obj_fcn)

        # name, labels, value of figure to be saved
        # all the figures are saved in the ''figs'' folder (make sure to create it)

        if settings.difference is False:
            #x_values, param = fig_paddle('mip_gaps', settings, folder)
            #plot_obj(all_mip_gaps[folder], x_values, param)

            x_values, param = fig_paddle('mip_times', settings, folder)
            print('Plotting sol times of %s. Values below ' % folder)
            plot_line(all_mip_times[folder], x_values, param)
        else:
            continue

    if settings.difference is True:
        if settings.cpp:
            plot_difference_cpp(settings, all_mip_gaps, all_mip_times, 'gaps')
        else:
            plot_difference(settings, all_mip_gaps, all_mip_times, 'gaps')

    if settings.together is True:
        plot_together(settings, all_mip_gaps, all_mip_times, 'gaps')






