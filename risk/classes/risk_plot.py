from matplotlib import pyplot as plt
import copy

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

    def plot_stat(self, stat_list=None, title=None, lgd=None, rate=True):

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

        ylabel = 'Rate [value/N]'
        if rate is False:
            y_label = 'Percentage [\%]'

        plt.xlabel('Instances [N]')
        plt.ylabel(ylabel)
        plt.show()

        self.clear_list()

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
    def assemble_parent_name(datename: str, config: str, probname=''):
        parent_name = datename + config + '-' + probname
        return parent_name

    @staticmethod
    def show_me():
        plt.show()
