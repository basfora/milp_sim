from matplotlib import pyplot as plt
from milp_sim.risk.classes.stats import MyStats
from milp_sim.risk.classes.danger import MyDanger
from milp_sim.risk.src import base_fun as bf
from milp_sim.risk.src.r_plotfun import CustomizePlot
import matplotlib.lines as mlines
from milp_sim.risk.src import plots_funs as pfs
import copy
import os
from scipy.stats import sem, t
from scipy import mean as mean_sp
import numpy as np

class RiskPlot:

    def __init__(self):

        self.n_runs = 1000
        self.list_runs = []

        self.customize = CustomizePlot()

        self.x_list = []
        self.y_list = []
        self.configs = 0
        self.fig_name = []

        self.avg = []
        self.std = []

        self.title = []
        self.lgd = []
        self.y_label = []
        self.compiled_files = []

        self.outcomes = []
        self.cum_outcomes = []

        self.casualties = []
        self.cum_casualties = []

        self.times = []
        self.cum_times = []

        self.stats_list = [self.outcomes, self.casualties, self.times]

        self.N = [i for i in range(1, self.n_runs + 1)]

    def clear_list(self):
        self.stats_list = []

    def set_compiled_files(self, my_list):
        self.compiled_files = my_list

    def add_stats(self, stat):

        if isinstance(stat, list):
            self.stats_list = stat
            n_runs = len(stat[0])

        else:
            self.stats_list.append(stat)
            n_runs = len(stat)
        self.set_n_turns(n_runs)

    def set_n_turns(self, value: int):
        self.n_runs = value
        self.set_run_list()

    def set_run_list(self):
        self.list_runs = list(range(1, self.n_runs + 1))
        self.x_list = list(range(1, self.n_runs + 1))
        return self.list_runs

    def plot_stat(self, stat_list=None, title=None, lgd=None, ylabel=0):

        if stat_list is not None:
            self.add_stats(stat_list)

        colors = ['k', 'r', 'b', 'y', 'm', 'gray']
        idx = -1
        for stat in self.stats_list:
            idx += 1
            self.y_list = stat
            # color
            my_color = colors[idx]
            #
            self.plot_points_from_list(self.x_list, self.y_list, my_color)

        if title is not None:
            plt.title(title)
        if lgd is not None:
            plt.legend(lgd)

        if ylabel == 0:
            ylabel = 'Rate [events/N]'
        elif ylabel == 1:
            ylabel = 'Percentage [\%]'
        else:
            ylabel = 'Time [time steps]'

        plt.xlabel('Instances [N]')
        plt.ylabel(ylabel)
        plt.show()

        self.clear_list()

    def retrieve_data(self, folder_name: str, instance_base: str, subtitle=''):
        """Retrieve data"""

        stat = MyStats(folder_name, instance_base)

        # rates to plot
        success_rate = stat.cum_success_rate
        failure_rate = stat.cum_fail_rate
        cutoff_rate = stat.cum_cutoff_rate
        casualty_rate = stat.cum_casualty_rate
        lgd = ['Success', 'Fail', 'Deadline Reached']
        title = 'Mission Status\n' + subtitle
        stats_plot = [success_rate, failure_rate, cutoff_rate]
        self.plot_stat(stats_plot, title, lgd, 1)

        avg_mission_time = stat.cum_mission_time
        avg_capture_time = stat.cum_capture_time
        lgd = ['Mission Time', 'Capture Time']
        title = 'Mission Times\n' + subtitle
        stats_plot = [avg_mission_time, avg_capture_time]
        self.plot_stat(stats_plot, title, lgd, 3)

        mission_casualties = stat.cum_casualty_mission_rate
        mva_casualties = stat.cum_casualty_mvp_rate
        non_mva_casualties = stat.cum_casualty_not_mvp_rate
        stats_plot = [mission_casualties, mva_casualties, non_mva_casualties]
        lgd = ['Missions with Casualties', 'MVA', 'Non-MVA']
        title = 'Casualties\n' + subtitle
        self.plot_stat(stats_plot, title, lgd, 0)

    # --------------------------------------
    # Retrieve for paper plots
    # --------------------------------------
    def retrieve_outcomes(self, pickle_names: list):

        outcomes_list = []
        outcomes_cum = []

        self.configs = len(pickle_names)

        for f_name in pickle_names:

            print('Getting data from %s' % f_name)
            out_data = []
            cum_data = []

            f_path = self.get_file_path(f_name)
            # load pickle file
            stat = bf.load_pickle_file(f_path)

            cum_data.append(stat.cum_success_rate)
            cum_data.append(stat.cum_fail_rate)
            cum_data.append(stat.cum_abort_rate)
            cum_data.append(stat.cum_cutoff_rate)

            out_data.append(stat.success_list)
            out_data.append(stat.fail_list)
            out_data.append(stat.abort_list)
            out_data.append(stat.cutoff_list)

            outcomes_list.append(out_data)
            outcomes_cum.append(cum_data)

        self.cum_outcomes = outcomes_cum
        self.outcomes = self.change_format_outcomes(self.N, outcomes_list)
        self.compute_MC(0)

        self.plot_error_point(0)

    def retrieve_times(self, pickle_names: list):

        times_list = []
        times_cum = []

        self.configs = len(pickle_names)

        for f_name in pickle_names:

            print('Getting data from %s' % f_name)
            out_data = []
            cum_data = []

            f_path = self.get_file_path(f_name)
            # load pickle file
            stat = bf.load_pickle_file(f_path)

            cum_data.append(stat.cum_mission_time)
            cum_data.append(stat.cum_abort_rate)
            cum_data.append(stat.cum_capture_time)

            out_data.append(stat.mission_time_list)
            out_data.append(stat.abort_time_list)
            out_data.append(stat.capture_time_list)

            times_list.append(out_data)
            times_cum.append(cum_data)

        self.cum_times = times_cum
        self.times = times_list # self.change_format_outcomes(self.N, times_list)
        self.compute_MC(2)

        self.plot_error_point(2)

    def retrieve_casualties(self, pickle_names: list):

        casualties_list = []
        casualties_cum = []

        plot_n = 1

        self.configs = len(pickle_names)

        for f_name in pickle_names:

            print('Getting data from %s' % f_name)
            list_data = []
            cum_data = []

            f_path = self.get_file_path(f_name)
            # load pickle file
            stat = bf.load_pickle_file(f_path)

            # actual data collection
            cum_data.append(stat.cum_casualty_mission_rate)
            cum_data.append(stat.cum_casualty_mvp_rate)
            cum_data.append(stat.cum_casualty_not_mvp_rate)

            list_data.append(stat.casualty_list)
            list_data.append(stat.casualty_mvp)

            casualties_list.append(list_data)
            casualties_cum.append(cum_data)

        self.cum_casualties = casualties_cum
        self.casualties = self.change_format_casualties(self.N, casualties_list)
        self.compute_MC(plot_n)

        self.plot_error_point(plot_n)

    def plot_error_point(self, plot_n=0):

        # get padding
        self.set_lgd(plot_n)
        self.set_x_ticks()
        self.set_title()

        fig = plt.figure()
        ax = fig.add_subplot(111)

        colors = ['bo', 'ro', 'ko', 'mo']

        for i in range(self.configs):

            for j in range(len(self.avg[i])):

                if plot_n == 0 and j == 1:
                    # do not plot
                    continue

                y = self.avg[i][j]
                std = self.std[i][j]

                if y is None:
                    continue

                # make error bar pretty
                low_error = std
                up_error = std

                if plot_n < 2:
                    max_y = 1.0
                else:
                    max_y = 100

                if y + std > max_y:
                    low_error = std
                    up_error = max_y - y
                elif y - std < 0.0:
                    up_error = std
                    low_error = y

                if plot_n < 2:
                    y = self.prob_to_per(y)
                    low_error = self.prob_to_per(low_error)
                    up_error = self.prob_to_per(up_error)

                assy = np.array([[low_error, up_error]]).T

                # make into lists
                x = [self.x_list[i]]
                y = [y]

                plt.errorbar(x, y, yerr=assy, fmt=colors[j])

        # plot labels
        f_size = 16
        ax.set_ylabel(self.y_label[plot_n], fontsize=f_size)

        # title and legend
        # plt.title(self.title[plot_n])

        my_handle = []
        for j in range(len(self.avg[0])):
            if plot_n == 0 and j == 1:
                continue

            my_handle.append(mlines.Line2D([], [], color=colors[j][0], label=self.lgd[j]))

        if plot_n == 1:
            my_loc = 'center left'
        elif plot_n == 0:
            my_loc = 'center left'
        else:
            my_loc = 'center left'
        plt.legend(handles=my_handle, frameon=False, loc=my_loc)

        # save fig
        fig_path = MyDanger.get_folder_path('figs')
        fig_name = fig_path + '/' + self.fig_name[plot_n] + '.pdf'
        fig.savefig(fig_name, bbox_inches='tight')

        # show me
        plt.show()

    # --------------------------------------
    # Computation of stats
    # --------------------------------------
    @staticmethod
    def change_format_outcomes(N, outcomes_list):

        config_outcomes = []
        for config in outcomes_list:
            binary_outcome = []
            for yes_list in config:
                aux_list = []

                for i in N:
                    if i in yes_list:
                        aux_list.append(1.0)
                    else:
                        aux_list.append(0.0)

                binary_outcome.append(aux_list)
            config_outcomes.append(binary_outcome)

        return config_outcomes

    @staticmethod
    def change_format_casualties(N, casualty_list):

        config_casual = []
        for config in casualty_list:

            binary_non_mva = []

            c_any = copy.copy(config[0])
            c_mva = copy.copy(config[1])

            # missions with casualties
            binary_any = [1 if c > 0 else 0 for c in c_any]
            print('--\nANY %.4f' % (sum(binary_any)/1000))
            # MVA
            binary_mva = [1 if c is True else 0 for c in c_mva]
            print('MVA %.4f' % (sum(binary_mva)/1000))

            for i in N:

                idx = i - 1

                # other
                if (binary_any[idx] == 1 and binary_mva[idx] == 0) or c_any[idx] > 1:
                    binary_non_mva.append(1.0)
                else:
                    binary_non_mva.append(0.0)
            print('N-MVA %.4f\n---' % (sum(binary_non_mva)/1000))
            config_casual.append([binary_any, binary_mva, binary_non_mva])

        return config_casual

    @staticmethod
    def prob_to_per(prob):
        per = round(prob * 100, 2)
        return per

    def compute_MC(self, plot_n=0):

        self.avg = []
        self.std = []

        if plot_n == 0:
            metric = self.outcomes
            cum_metric = self.cum_outcomes
        elif plot_n == 1:
            metric = self.casualties
            cum_metric = self.cum_casualties
        elif plot_n == 2:
            metric = self.times
            cum_metric = self.cum_times

        else:
            return

        i = -1
        for config in metric:
            i += 1
            j = -1

            avg_config = []
            std_config = []
            for rate in config:
                j += 1
                avg, std = self.compute_avg(rate)

                if not isinstance(avg, float) and not isinstance(avg, int):
                    avg = None
                    std = None

                avg_config.append(avg)
                std_config.append(std)

                if self.sanity_check(avg, cum_metric[i][j][-1]) is False:
                    print('config = %d, metric = %d' % (i, j))

            self.avg.append(avg_config)
            self.std.append(std_config)

    @staticmethod
    def sanity_check(avg1, avg2):

        if avg1 is None or avg2 is None:
            return False

        if round(avg1, 2) == round(avg2, 2):
            return True
        else:
            print('Difference: computed %.4f, cumulative %.4f ' % (avg1, avg2))
            return False

    @staticmethod
    def compute_avg(my_list: list):

        avg = np.mean(my_list)
        std = np.std(my_list)

        return avg, std

    # --------------------------------------
    # Plot padding
    # --------------------------------------
    def set_title(self):

        self.title.append('Mission Outcome')
        self.title.append('Missions with Casualties')
        self.title.append('Mission Times')

        self.y_label = ['Percentage of Missions [\%]', 'Percentage of Missions [\%]', 'Time [steps]']

        self.fig_name = ['mission_outcomes', 'casualties', 'times']

    def set_config_names(self):

        ND = 'No danger baseline'
        PK = 'A priori: true values'
        I100 = 'A priori: uniform, estimation: 100\% images'
        I5 = 'A priori: uniform, estimation: 5\% images'
        NC = 'No danger constraints'

        self.lgd = [ND, PK, I100, I5, NC]

    def set_lgd(self, n_plot=0):
        if n_plot == 0:
            self.lgd = ['Success', 'Failure', 'Abort', 'Cutoff']
        elif n_plot == 1:
            self.lgd = ['Any', 'MVA', 'Other']
        elif n_plot == 2:
            self.lgd = ['Average', 'Abort', 'Capture']

    def set_x_ticks(self):

        ND = 'No danger baseline'
        PK = 'A priori: true values'
        I100 = 'A priori: uniform, estimation: 100\% images'
        I5 = 'A priori: uniform, estimation: 5\% images'
        NC = 'No danger constraints'

        self.x_list = ['ND', 'PK', 'I100', 'I5', 'NC']

    # --------------------------------------
    # File path
    # --------------------------------------
    @staticmethod
    def get_compiled_path():
        data_compiled_path = MyDanger.get_folder_path('data_compiled')
        save_path = data_compiled_path
        return save_path

    def get_file_path(self, f_name: str):
        f_path = self.get_compiled_path() + '/' + f_name + '.pkl'
        return f_path

    @staticmethod
    def assemble_parent_name(datename: str, config: str, extra_id='', probname='h'):

        parent_name = datename + config + '-' + probname + '-' + extra_id
        data_saved_path = MyDanger.get_folder_path('data_saved')

        parent_path = data_saved_path + '/' + parent_name
        # check if folder exists
        if not os.path.exists(parent_path):
            exit(print('Parent folder %s does not exist.' % parent_name))

        return parent_name, parent_path

    # --------------------------------------
    # General functions
    # --------------------------------------
    @staticmethod
    def plot_points_between_list(v_points, v_conn, my_color='k', my_marker=None):
        """Plot points and their connections
        :param v_points = [(x1, y1), (x2, y2)...]
        :param v_conn = [(0, 1), (1, 2)...]
        :param my_color
        :param my_marker"""

        my_handle = None
        for edge in v_conn:
            i0 = edge[0]
            i1 = edge[1]

            n0 = v_points[i0]
            n1 = v_points[i1]

            px = [n0[0], n1[0]]
            py = [n0[1], n1[1]]

            marker_size = 3

            my_handle = plt.plot(px, py, color=my_color, marker=my_marker, markersize=marker_size,
                                 linestyle='solid', linewidth=2)

        return my_handle

    @staticmethod
    def plot_points(vertices: dict or list, my_color='k', my_marker='o', sizemarker=2):
        """Plot vertices from
         (dict) V[v] = (x,y)
         (list) V = [(x1, y1), (x2, y2)...]"""

        my_handle = None
        if isinstance(vertices, dict):
            for k in vertices.keys():
                if not isinstance(k, int):
                    continue
                x = [vertices[k][0]]
                y = [vertices[k][1]]
                my_handle = plt.plot(x, y, color=my_color, marker=my_marker, markersize=sizemarker)
        elif isinstance(vertices, list):
            for v in vertices:
                x = v[0]
                y = v[1]
                my_handle = plt.plot(x, y, color=my_color, marker=my_marker, markersize=sizemarker)
        else:

            print('Wrong input format, accepts dict or list')

        return my_handle

    @staticmethod
    def plot_points_from_list(x_list: list, y_list: list, my_color='k', my_marker='.'):

        plt.scatter(x_list, y_list, color=my_color, marker=my_marker, linewidth=1)

        return

    @staticmethod
    def show_me():
        plt.show()

    # @staticmethod
    # def draw_errorbar(xvalues, yvalues, y_err, ax, l_style='bo-', lgd=None):
    #     ax.errorbar(xvalues, yvalues, yerr=y_err, fmt=l_style, label=lgd, capsize=2)
    #     ax.grid(b=True, which='both')