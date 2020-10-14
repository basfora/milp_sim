import os
from igraph import plot
from matplotlib import pyplot as plt
from milp_mespp.core import extract_info as ext
from math import sqrt
from milp_sim.risk.src import base_fun as bf
from milp_sim.risk.classes.danger import MyDanger
from milp_sim.scenes import make_graph as mg, files_fun as ff
from milp_sim.risk.src import r_plotfun as rpf
from collections import Counter


class RiskPlot:

    def __init__(self):

        self.handle = None
        self.graph_vertices = 'School_Image_VGraph'
        self.graph_edges = 'School_Image_EGraph'

        self.f_name = 'saved_data.pkl'
        self.n = 46

        self.danger_files = []
        self.eta_values = dict()
        self.z_values = dict()

        self.levels, self.n_levels, self.level_label, self.level_color = MyDanger.define_danger_levels()

        # data
        # to be updated for each parent folder
        self.parent = ''
        self.belief, self.target, self.team, self.solver_data, self.danger, self.mission, self.specs = self.empty_dicts(7)
        self.percentages = []
        self.configs = []
        self.n_runs = 200
        self.instances = []

        # folders
        self.parent_folders = []
        # parent folders as keys
        self.instance_folders = dict()

        # stats
        self.casualties = dict()
        self.mission_time = dict()
        self.killed_info = dict()
        self.success = dict()
        self.failure = dict()
        self.deadline_reached = dict()
        self.team_killed = dict()
        self.v_sort = dict()
        self.z_sort = dict()

        self.v_fatal = dict()
        self.z_fatal = dict()

        self.v_bleak = []
        self.v_times = []
        self.z_cause = []
        self.k_team = []
        self.casualties_kappa = dict()

        self.avg_time = []

        self.prob_kill = []

    @staticmethod
    def empty_dicts(q=7):
        my_dicts = (dict() for i in range(q))
        return my_dicts

    def set_danger_files(self, list_files=None):
        """Name of source files for danger data"""

        if list_files is None:
            list_files = ['danger_map_NCF_freq_05', 'danger_map_NCF_freq_100']

        self.danger_files = list_files
        self.load_eta_values()
        self.set_z_values()
        # self.print_danger_data()
        self.set_configs_per_img()

    def set_configs_per_img(self):
        self.percentages = [per for per in self.z_values]

        self.casualties = {per: 0 for per in self.percentages}
        self.mission_time = {per: [] for per in self.percentages}
        self.killed_info = {per: [] for per in self.percentages}
        self.success = {per: 0 for per in self.percentages}
        self.failure = {per: 0 for per in self.percentages}
        self.deadline_reached = {per: 0 for per in self.percentages}
        self.team_killed = {per: 0 for per in self.percentages}

        self.v_fatal = {per: [] for per in self.percentages}
        self.z_fatal = {per: [] for per in self.percentages}

        self.v_sort = {per: [] for per in self.percentages}
        self.z_sort = {per: [] for per in self.percentages}

    def load_eta_values(self):
        """load the actual values from file"""

        for f_name in self.danger_files:
            per = int(f_name.split('_')[-1])
            self.eta_values[per] = bf.is_list(MyDanger.load_danger_data(f_name))

    def set_z_values(self):

        for per in self.eta_values.keys():
            eta = self.eta_values.get(per)
            z = MyDanger.all_z_from_eta(eta, 1)
            self.z_values[per] = z

    def print_danger_data(self):

        for v_idx in range(self.n):
            print_str = 'v = ' + str(v_idx + 1)
            for per in self.z_values.keys():
                z = self.z_values.get(per)[v_idx]
                my_str = ', z_' + str(per) + '= ' + str(z)
                print_str += my_str
            print(print_str)

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

        if isinstance(vertices, dict):
            for k in vertices.keys():
                if not isinstance(k, int):
                    continue
                x = [vertices[k][0]]
                y = [vertices[k][1]]
                plt.plot(x, y, color=my_color, marker=my_marker, markersize=sizemarker)
        elif isinstance(vertices, list):
            for v in vertices:
                x = v[0]
                y = v[1]
                plt.plot(x, y, color=my_color, marker=my_marker, markersize=sizemarker)
        else:
            print('Wrong input format, accepts dict or list')

        return None

    def plot_graph_school(self):

        bloat = True
        per_list = [per for per in self.z_values.keys()]
        n_sub = len(per_list)

        for i in range(n_sub):
            fig_1, ax_arr = plt.subplots(1, 1, figsize=(9, 5), dpi=150)

            mg.plot_ss2(bloat)

            mg.plot_graph(bloat, True, True)

            V = ff.get_info(self.graph_vertices, 'V')

            danger = True

            if danger:

                colors = self.get_vertices_color(per_list[i])
                for v in V:
                    vidx = V.index(v)
                    my_color = colors[vidx]

                    self.plot_points([v], my_color, 'o', 6)

            xy = [0.2, 0.15]
            rpf.my_hazard_labels(fig_1, xy, 16)

            my_str = 'Danger Graph, ' + str(per_list[i]) + '\% images'
            plt.title(my_str)

            fig_name = 'danger_graph_' + str(per_list[i])
            mg.save_plot(fig_name, 'figs', '.png')

            # plt.show()

    def get_vertices_color(self, per=100):
        my_color = []
        for v_idx in range(self.n):
            z_v = self.z_values[per][v_idx]
            color_v = rpf.match_level_color(z_v)
            my_color.append(color_v)

        return my_color

    def set_path_data(self, parent_folder='10-07-2'):

        saved_data_path = MyDanger.get_folder_path('data_saved')

        self.parent = saved_data_path + '/' + parent_folder

    def collect_data(self, parent_folder='10-07-2', start_name='point', date_name='1008'):

        self.set_path_data(parent_folder)

        middle_name = '_G46V_ss2_' + date_name + '_'

        f_name = 'saved_data'
        extension = 'pkl'

        for per in self.percentages:

            for n_file in self.instances:
                folder = start_name + str(per) + middle_name + str(n_file).zfill(3)

                temp_path = self.parent + '/' + folder
                f_path = bf.assemble_file_path(temp_path, f_name, extension)

                data = bf.load_pickle_file(f_path)

                self.belief[per, n_file] = data['belief']
                self.target[per, n_file] = data['target']
                self.danger[per, n_file] = data['danger']
                try:
                    self.mission[per, n_file] = data['mission']
                except:
                    print('No mission obj')
                self.team[per, n_file] = data['team']
                self.solver_data[per, n_file] = data['solver_data']
                self.specs[per, n_file] = data['specs']

        self.prob_kill = [round(l * 100, 2) for l in self.danger[per, n_file].prob_kill]
        self.k_team = self.team[per, n_file].kappa

    def get_stats(self):

        # loop through percentages
        for per in self.percentages:
            print('Collecting data from %d percent of images' % per)

            self.casualties_kappa[per] = [0, 0, 0]

            # loop through instances
            for i in self.instances:
                # get mission info
                mission = self.mission[per, i]
                team = self.team[per, i]

                # avg mission time
                self.mission_time[per].append(mission.last_t)

                # success or failure
                if mission.target_capture:
                    self.success[per] += 1
                else:
                    if mission.deadline_reached:
                        self.deadline_reached[per] += 1
                    elif mission.team_killed:
                        self.team_killed[per] += 1
                    self.failure[per] += 1

                # casualties
                n_killed = len(team.killed)
                self.casualties[per] += n_killed
                if n_killed > 0:
                    for s in team.killed_info.keys():
                        v = team.killed_info.get(s)[0]
                        t = team.killed_info.get(s)[1]
                        z = team.killed_info.get(s)[2]
                        k = team.killed_info.get(s)[3]

                        self.killed_info[per].append((v, t, z, k))
                        self.v_fatal[per].append(v)
                        self.z_fatal[per].append(z)
                        idx = self.k_team.index(k)
                        self.casualties_kappa[per][idx] += 1

            self.v_sort[per] = self.killing_spot(self.v_fatal[per])
        self.killing_danger()
        self.print_stats()

    @staticmethod
    def assemble_parent_name(datename: str, config: str, probname=''):
        parent_name = datename + config + '-' + probname
        return parent_name

    @staticmethod
    def killing_spot(v_list: list):

        v_trouble = Counter(v_list)
        v_out = []
        for v in v_trouble.keys():
            v_out.append((v, v_trouble[v]))

        return v_out

    def killing_danger(self):

        v_bleak = []
        v_times = []
        for per in self.percentages:
            for vx in self.v_sort[per]:
                v = vx[0]
                x = vx[1]
                if v not in v_bleak:
                    v_bleak.append(v)
                    v_times.append(x)
                idx = v_bleak.index(v)
                v_times[idx] += x

        z_cause = []
        for v in v_bleak:
            z_cause.append(self.danger[100, 1].get_z(v))

        self.v_bleak = v_bleak
        self.v_times = v_times
        self.z_cause = z_cause

    def print_stats(self):

        percentage_success = []

        for per in self.percentages:
            print('---\n')
            print('Stats for %d percent of images: ' % per)

            # ---------------------
            # success and failures
            # ---------------------

            success = self.success[per]
            suc_per = round(100 * success/self.n_runs, 2)
            percentage_success.append(suc_per)
            failures = self.failure[per]
            fail_per = round(100 * failures/self.n_runs, 2)

            print('Mission: success =  %d (%d/100),  failure = %d (%d/100) between team killed: %d and deadline reached: %d'
                  % (success, suc_per, failures, fail_per, self.team_killed[per], self.deadline_reached[per]))

            # ---------------------
            # casualties and mission time
            # ---------------------

            print('---')
            avg_time = round(sum(self.mission_time[per])/len(self.mission_time[per]))
            self.avg_time.append(avg_time)
            print('Average mission time: % d' % avg_time)
            print('Casualties: %d/%d ' % (self.casualties[per], 3 * self.n_runs))
            print('Casualties per threshold: ' + str(self.casualties_kappa[per]) + ' for kappa = ' + str(self.k_team))

        # ---------------------
        # Killing spots
        # ---------------------
        print('---\n---')
        print('Fatal vertices: ' + str(self.v_bleak))
        print('Number of casualties: ' + str(self.v_times))
        print('True danger levels: ' + str(self.z_cause))
        print('Probability kill: ' + str(self.prob_kill))

        # easier to see
        print('\nOrder (summary): ' + str(self.percentages))
        c = [round(100 * self.casualties[per]/(3*self.n_runs), 2) for per in self.percentages]
        print('Casualties: ' + str(c))
        k = [self.casualties_kappa[per][0] for per in self.percentages]
        print('Casualties of lowest threshold: ' + str(k))
        print('Mission Times: ' + str(self.avg_time))
        print('Mission success: ' + str(percentage_success))


if __name__ == "__main__":

    risk_data = RiskPlot()
    risk_data.set_danger_files()
    risk_data.print_danger_data()
    risk_data.plot_graph_school()
    #
    # parent_folder = '10-09-NC-NK-2'
    # start_name = 'NC_NK_point'
    # date_name = '1009'
    # #
    # risk_data.collect_data(parent_folder, start_name, date_name)
    # risk_data.get_stats()
