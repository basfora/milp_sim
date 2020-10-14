from matplotlib import pyplot as plt
from milp_sim.risk.classes.stats import MyStats
from milp_sim.risk.classes.danger import MyDanger
import copy
import os


class RiskPlot:

    def __init__(self):

        self.stats_list = []
        self.n_runs = 0
        self.list_runs = []

        self.x_list = []
        self.y_list = []

    def clear_list(self):
        self.stats_list = []

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
    def assemble_parent_name(datename: str, config: str, extra_id='', probname='h'):

        parent_name = datename + config + '-' + probname + '-' + extra_id
        data_saved_path = MyDanger.get_folder_path('data_saved')

        parent_path = data_saved_path + '/' + parent_name
        # check if folder exists
        if not os.path.exists(parent_path):
            exit(print('Parent folder %s does not exist.' % parent_name))

        return parent_name, parent_path

    @staticmethod
    def show_me():
        plt.show()

