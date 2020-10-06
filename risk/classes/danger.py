"""All functions related to danger computation"""
import random
from milp_mespp.core import data_fun as df, extract_info as ext
from milp_sim.risk.src import base_fun as bf
import copy


class MyDanger:
    """Define danger
    OBS: for ICRA simulations danger is constant for planning horizon"""

    def __init__(self, g, eta_true, etahat_offline=None, eta0_0=None):
        """
        :param g : graph
        :param eta_true : ground truth danger level for each vertex
        :param eta0_0 : a priori probability of danger
        :param plot_for_me
         """

        # ---------------------
        # pre defined parameters
        # ---------------------
        # danger levels
        self.levels, self.n_levels, self.level_label, self.level_color = self.define_danger_levels()
        self.options = self.define_options()

        # -----------
        # input parameters (graph, immutable)
        # -----------

        # save graph
        self.g_name = g['name']
        self.g = g
        # vertices
        self.V, self.n = ext.get_set_vertices(g)
        self.fov = self.get_fov_ss2()

        # ----------
        # team of searchers (get rid of this to avoid issues when searcher is dead?)
        # ----------
        self.S, self.m = [], 0
        # danger threshold for the searchers
        self.kappa = list()
        self.alpha = list()

        # ground truth
        # eta = [eta_l1, ... eta_l5]
        self.eta = []
        # z = argmax eta_l \in {1,..5}
        self.z = []
        self.H = []

        # probability kill
        self.prob_kill = []
        self.set_prob_kill()

        # ------------
        # estimation
        # ------------

        # a priori knowledge of danger
        # eta0_0[v] = [eta0_0_l1,... eta0_0_l5]
        self.eta0_0 = []
        # z0_0[v] = level, level in {1,...5}
        self.z0_0 = []

        # estimates at each time step
        # eta_hat[v] = [l1,...l5]
        self.eta_hat = []
        # probable danger level
        self.z_hat = []

        # for each searcher
        # H[s] = [H_s1, H_s2...H_sm] (estimate)
        self.H_hat = []
        self.H_level_hat = []

        # to make getting the values easier
        # matching scores xi[v] = [[i1_1...i1_5], [i2_1,...i2_5]]
        self.xi = dict()
        # look up of estimated danger for each vertex (when robots see every image in that vertex)
        self.lookup_eta_hat = []
        # H_l[v] = [l1...l5] (true value)
        self.lookup_H_level = []
        self.lookup_z_hat = []

        # -------------
        # storage
        # -------------
        # probability of each danger level: eta_hat[v,t] = [prob l1, ...l5]
        self.stored_eta_hat = {0: list()}
        # probable danger level, argmax eta: eta_hat[v,t] = level
        self.stored_z_hat = {0: list()}
        # cumulative danger, from eta_hat
        self.stored_H_hat = {0: list()}
        # name of folder for this sim: eg smoke_G9V_grid_date#_run#
        self.folder_name = ""
        # whole path + /name_folder
        self.path_exp_data = self.get_folder_path('exp_data')

        # -----------------------
        # danger data for simulation (files)
        # -----------------------
        # true distributions (100% images)
        self.true_file_name = ''
        self.true_file_path = ''
        self.true_raw = None
        # estimated offline (% images)
        self.estimated_file_name = ''
        self.estimated_file_path = ''
        self.percentage_im = 0.25
        self.estimated_raw = None
        # path to files
        self.folder_path = self.get_folder_path()
        # ----------------------
        # matching scores
        self.set_scores('node_score_dict_Fire.p')

        # set up danger values
        # ground truth
        extension = 'pkl'
        if isinstance(eta_true, str):
            self.gnd_truth_from_file(eta_true, extension)
        else:
            # useful for unit tests and sanity checks
            self.gnd_truth_from_value(eta_true)

        # a priori
        self.set_priori(eta0_0)

        # estimated danger
        if isinstance(etahat_offline, str):
            # TODO function to retrieve this value from file name
            percentage = 0.25
            self.estimated_from_file(etahat_offline, percentage, extension)
        else:
            self.estimated_from_value(etahat_offline)

        # point or prob (default: point)
        self.perception = self.options[0]

    # ----------------------
    # From files --- load danger data and get ready for simulation
    # ----------------------

    def gnd_truth_from_file(self, f_true: str, extension='pkl'):
        """Danger data (true): set files, load and format to simulation"""

        # point to file
        self.set_true_file(f_true, extension)
        # load, save and format self.eta
        self.load_data_from_file('true')

    def set_true_file(self, f_name: str, extension='pkl'):
        """Set file where to get danger data (true value - 100 % images)"""
        self.true_file_name = f_name
        self.true_file_path = bf.assemble_file_path(self.folder_path, f_name, extension)

    def load_data_from_file(self, op='true'):
        """Load raw data and save it"""

        if op == 'true':
            eta_true = bf.load_pickle_file(self.true_file_path)
            self.true_raw = eta_true
            self.eta = bf.is_list(self.true_raw)
            # compute z
            self.z = self.all_z_from_eta(self.eta)

        elif op == 'hat':
            eta_hat = bf.load_pickle_file(self.estimated_file_path)
            self.estimated_raw = eta_hat
            # set lookup table
            etahat_list = bf.is_list(self.estimated_raw)
            self.set_lookup_hat(etahat_list)

        return

    def estimated_from_file(self, f_name: str, percentage: float, extension='pkl'):
        # point to file
        self.set_estimated_file(f_name, percentage, extension)
        # load, save and format self.lookup_eta
        self.load_data_from_file('hat')

    def set_estimated_file(self, f_name: str, percentage: float, extension='pkl'):
        """Set file where to get danger data (estimated, 25, 50 or 75%)"""
        self.estimated_file_name = f_name
        self.percentage_im = percentage
        self.estimated_file_path = bf.assemble_file_path(self.folder_path, f_name, extension)

    # -----------------
    # for direct input of a priori / true data
    # -----------------
    # UT - ok
    def set_priori(self, eta_0):

        n = self.n

        eta0_0, z0_0 = self.compute_from_value(n, eta_0)

        # eta_priori[v] = [eta_l1,... eta_l5]
        self.eta0_0 = eta0_0

        H_levels = []
        for v in range(self.n):
            H_v = self.compute_H(eta0_0[v])
            H_levels.append(H_v)
        self.H_level_hat = H_levels

        # eta_priori_hat[v] = [level], level in {1,...5}
        self.z0_0 = z0_0

        # first estimate is a priori
        self.eta_hat = eta0_0
        self.z_hat = z0_0

        # save in storage
        self.stored_eta_hat[0] = eta0_0
        self.stored_z_hat[0] = z0_0

        return

    # UT - ok
    def gnd_truth_from_value(self, eta_true):

        eta, z = self.compute_from_value(self.n, eta_true)

        # eta = [eta_l1,... eta_l5]
        self.eta = eta
        self.z = z

    def estimated_from_value(self, my_eta_hat):
        """Mostly for testing purposes"""

        z_hat = None

        if my_eta_hat is None:
            # make it equal to gnd truth
            eta_hat = copy.copy(self.eta)
            z_hat = copy.copy(self.z)
        elif my_eta_hat == 0:
            eta_hat = copy.copy(self.eta0_0)
            z_hat = copy.copy(self.z0_0)
        elif my_eta_hat == 1:
            # frequentist (just for testing), computing from matching scores
            xi = self.xi
            eta_hat, z_hat = self.compute_frequentist(xi)
        else:
            eta_hat = bf.is_list(my_eta_hat)

        self.set_lookup_hat(eta_hat, z_hat)

    def set_lookup_hat(self, eta_hat, z_hat=None):
        """Pre-Compute the estimated eta and z for all vertices,
         considering robot is at that vertex"""

        # distribution
        self.lookup_eta_hat = eta_hat

        # point estimate
        if z_hat is None:
            # compute z
            self.lookup_z_hat = self.all_z_from_eta(self.lookup_eta_hat)
        else:
            self.lookup_z_hat = z_hat

        # cumulative distribution
        self.lookup_H_level = self.compute_all_H(eta_hat)

    def set_scores(self, xi: str or dict):

        if isinstance(xi, str):
            my_xi = MyDanger.load_scores(xi)
        else:
            my_xi = xi

        self.xi = my_xi

    def set_fov(self, op_test=False):

        if op_test is False:
            self.fov = self.get_fov_ss2()
        else:
            self.fov = self.get_fov_9g()

    # -----------------
    # set or update parameters
    # -----------------

    def set_perception(self, option: int or str):
        """Set if point or probability estimate
        option = 1 : point, 2 : prob"""
        if isinstance(option, int):
            self.perception = self.options[option]

        elif isinstance(option, str):
            if option not in self.options:
                print('Error! Danger type not valid, please choose from %s' % str(self.options))
                exit()

            self.perception = option

    def update_teamsize(self, m: int):
        self.m = m
        self.S = [i for i in range(1, m + 1)]

    def set_thresholds(self, kappa=3, alpha=0.95):
        k, a = [], []

        if isinstance(kappa, int):
            if self.m < 1:
                print('Please update team size.')
                exit()
            else:
                for s in self.S:
                    k.append(kappa)
                    a.append(alpha)

        elif isinstance(kappa, list):
            k, a = kappa, alpha
            self.update_teamsize(len(k))

        else:
            print('Provide integer or list')
            exit()

        self.kappa = k
        self.alpha = a

    def save_estimate(self, eta_hat=None, z_hat=None, H_hat=None, t=None):
        """Save estimated values for time step"""

        if t is None:
            t = ext.get_last_key(self.stored_z_hat)

        if eta_hat is None:
            eta_hat = copy.copy(self.eta_hat)
            z_hat = copy.copy(self.z_hat)
            H_hat = copy.copy(self.H_hat)

        # stored_z_hat[t] = [z_v1, z_v2....z_vn]
        self.stored_z_hat[t] = z_hat
        # stored_eta_hat[t] = [eta_v1,..., eta_vn], eta_v1 = [eta_1, ...eta_5]
        self.stored_eta_hat[t] = eta_hat
        self.stored_H_hat[t] = H_hat

    def update_estimate(self, eta_hat, z_hat, H_hat=None):

        self.eta_hat = eta_hat
        self.z_hat = z_hat
        self.H_hat = H_hat

    def estimate_simple(self):
        # TODO modify this

        my_hat, my_H = [], []

        for v in range(self.n):
            if self.perception == self.options[0]:
                hat_v = self.lookup_z_hat
            else:
                hat_v = self.lookup_eta_hat
                H_v = self.compute_H(my_hat, self.kappa)
                my_H.append(H_v)
            my_hat.append(hat_v)

        if self.perception == self.options[0]:
            self.z_hat = my_hat
        else:
            self.eta_hat = my_hat
            self.H_hat = my_H

    # UT - ok
    def estimate(self, visited_vertices):
        """Estimate danger level based on:
        current estimate + current position (new measurements)
        FOV of current vertices
        a priori (for non-visited/non-line of sight vertices)
        """

        # line of sight
        fov = self.fov

        # new update
        eta_hat = []
        z_hat = []

        # loop through vertices
        for v in self.V:
            # get priori (if no info available, it will keep it)
            etahat_v = self.get_etahat(v)
            zhat_v = self.get_zhat(v)

            # if vertex was visited, get estimated from lookup table
            if v in visited_vertices:
                etahat_v, zhat_v = self.get_from_lookup(v)

            # if not, check if is in line of sight of a visited vertex (closest one)
            else:
                sight_list = fov[v]
                # if list is not empty
                if len(sight_list) > 0:
                    # sort by distance (closest to further)
                    sight_list.sort(key=lambda k: [k[1], k[0]])

                    for neighbor in sight_list:
                        # check if this vertex u was visited
                        u = neighbor[0]

                        if u not in visited_vertices:
                            continue
                        else:
                            # if it was, set the estimate for this vertex equal to its neighbor
                            etahat_v, zhat_v = self.get_from_lookup(u)
                            break

            # append to estimated list
            eta_hat.append(etahat_v)
            z_hat.append(zhat_v)

        self.update_estimate(eta_hat, z_hat)
        self.save_estimate()

    # -------------------
    # Casualty functions
    # -------------------
    def set_prob_kill(self, prob_list=None):
        """Set the probability of killing a robot w.r.t. to true danger level
        Default:
        other options: prob_kill = [0.025, 0.05, 0.075, 0.1, 0.125]
        prob_kill = [0.1, 0.2, 0.3, 0.4, 0.5]
        prob_kill = [0.05, 0.1, 0.15, 0.2, 0.25]"""

        if prob_list is None:
            prob_kill = [0.025, 0.05, 0.075, 0.1, 0.125]
        else:
            prob_kill = prob_list

        self.prob_kill = prob_kill

    def is_fatal(self, v: int):
        """Determine if robot was killed by danger
        based on p(kill) for the true level of danger in vertex v
        return true (was killed) or false (not killed)"""

        level = self.get_z(v)

        # draw your luck
        is_fatal = self.draw_prob_kill(self.prob_kill, level)

        return is_fatal

    @staticmethod
    def draw_prob_kill(prob_list: list, level: int):
        """Draw from danger prob(kill) to see if it was a higher number (killed) or lower (not killed)"""

        l_idx = level - 1
        prob_kill = prob_list[l_idx]
        status = False

        # flip a coin
        robot_chance = random.random()

        if robot_chance <= prob_kill:
            status = True

        return status

    # ----------------------
    # Retrieve values (do not change class)
    # ----------------------
    # UT - ok
    def get_z(self, v: int):
        # get level for that vertex (z_true)
        v_idx = ext.get_python_idx(v)
        true_level = self.z[v_idx]

        return true_level

    def get_eta(self, v: int):
        # get level for that vertex (z_true)
        v_idx = ext.get_python_idx(v)
        eta = self.eta[v_idx]

        return eta

    # UT - ok
    def get_zhat(self, v: int):
        # get level for that vertex (z_hat)
        v_idx = ext.get_python_idx(v)
        level_hat = self.z_hat[v_idx]

        return level_hat

    def get_etahat(self, v: int):
        # get level for that vertex (eta_hat)
        v_idx = ext.get_python_idx(v)
        eta_hat = self.eta_hat[v_idx]

        return eta_hat

    def get_from_lookup(self, v: int):
        # get level for that vertex (eta_hat)
        v_idx = ext.get_python_idx(v)
        eta_hat = self.lookup_eta_hat[v_idx]
        z_hat = self.lookup_z_hat[v_idx]

        return eta_hat, z_hat

    def get_vertex_hat(self, v: int):
        """Retrieve estimated danger for that vertex
        if perception point -- retrieve z
        if perception prob -- retrieve eta"""

        v_idx = ext.get_python_idx(v)

        if self.perception == self.options[0]:
            # point
            my_hat = self.lookup_z_hat[v_idx]
        else:
            # probabilistic
            my_hat = self.lookup_eta_hat[v_idx]

        return my_hat

    # UT - ok
    @staticmethod
    def sum_1(eta: list):
        sum_prob = sum(eta)

        if 1 - 0.9 < sum_prob < 1 + 1.1:
            return True
        else:
            return False

    # UT - ok
    @staticmethod
    def compute_from_value(n, my_eta=None):
        """Compute initial distribution
        :param n : vertices number |V|
        :param my_eta : int, list or None
        if no argument: set uniform probability
        """
        eta0_0, z0_0 = [], []
        op = 2

        if my_eta is None:
            print('Setting uniform probability')
            for v in range(n):
                eta0_0.append([0.2 for i in range(5)])
                z0_0.append(3)

        elif isinstance(my_eta, int):
            for v in range(n):
                eta0_0.append(MyDanger.eta_from_z(my_eta))
                z0_0.append(my_eta)

        elif isinstance(my_eta, list):
            # list of lists (prob for each vertex)
            if isinstance(my_eta[0], list):
                eta0_0 = my_eta
                for v in range(n):
                    z0_0.append(MyDanger.z_from_eta(eta0_0[v], op))
            # list of danger for each vertex
            else:
                z0_0 = my_eta
                for v in range(n):
                    eta0_0.append(MyDanger.eta_from_z(z0_0[v]))

        else:
            print('Wrong input for a priori danger values')
            exit()

        return eta0_0, z0_0

    # --------------------
    # distribution to point estimate and vice versa
    # --------------------

    @staticmethod
    def all_z_from_eta(eta: list or dict):
        """Return list of point estimate for all vertices"""

        eta = bf.is_list(eta)

        n = len(eta)
        z = []

        for vidx in range(n):
            eta_v = eta[vidx]
            z_v = MyDanger.z_from_eta(eta_v, 1)
            z.append(z_v)

        return z

    # UT - ok
    @staticmethod
    def eta_from_z(my_level: int):
        """return list"""
        # TODO change if necessary (discrete probability)
        L = [1, 2, 3, 4, 5]
        max_l = 0.6
        o_l = round((1 - max_l)/4, 4)

        eta = []
        for level in L:
            if level == my_level:
                eta.append(max_l)
            else:
                eta.append(o_l)

        return eta

    # UT - ok
    @staticmethod
    def z_from_eta(eta_v: list, op=1, k=None):
        """
        op 1: conservative (max value)
        op 2: # min threshold for team """

        z = 1
        z_list = MyDanger.argmax_eta(eta_v)

        if op == 1:
            # get maximum prob, if tie pick the one closest to the middle
            z = MyDanger.pick_weighted_avg_z(z_list, eta_v)
        elif op == 2:
            # get maximum prob, if tie pick conservative (max value)
            z = MyDanger.pick_conservative_z(z_list)
        elif op == 3:
            # get max prob, if tie pick min threshold for team (break ties)
            z = MyDanger.pick_z_from_kappa(z_list, k)
        else:
            exit(print('No other options!'))

        return z

    # UT - ok
    @staticmethod
    def argmax_eta(eta_v: list):
        """Get list of maximum values (might be just one element)
        return list with at least one element"""

        z_list = [idx + 1 for idx, val in enumerate(eta_v) if val == max(eta_v)]

        return z_list

    # UT - ok
    @staticmethod
    def pick_weighted_avg_z(z_list: list, eta_v: list):
        """z = max prob level (mean if a tie)
        eta = [eta_l1,... eta_l5]
        return argmax occurrence, if tie return closest to mean
        return integer"""
        levels = [1, 2, 3, 4, 5]

        if len(z_list) > 1:
            weight = [eta_v[i] * levels[i] for i in range(5)]
            my_sum = sum(weight)
            z = round(my_sum)
        else:
            z = z_list[0]

        return z

    # UT - ok
    @staticmethod
    def pick_conservative_z(z_list: list):
        """z = max prob level (conservative!)
        eta = [eta_l1,... eta_l5]
        return last occurrence (max danger value)
        return integer"""

        z = max(z_list)

        return z

    # UT - ok
    @staticmethod
    def pick_z_from_kappa(z_list: list, kappa: list):
        """eta_hat = max prob level
        eta = [eta_l1,... eta_l5]
        return last occurrence (max danger value)
        return integer"""

        # min team threshold
        k_min = min(kappa)

        z = 1
        for my_z in z_list:
            if my_z > k_min:
                break
            z = my_z

        return z

    @staticmethod
    def compute_all_H(eta: list, kappa=None):

        n = len(eta)
        H = []

        for vidx in range(n):
            eta_v = eta[vidx]
            H_v = MyDanger.compute_H(eta_v, kappa)
            H.append(H_v)

        return H

    # UT - ok
    @staticmethod
    def compute_H(eta_v: list, kappa=None):
        """H = sum_1^l eta_l
        return list
        :param eta_v : prob of levels 1,...5
        :param kappa : list of thresholds, if none do for levels
        """

        H = []

        if kappa is None:
            k_list = [l for l in range(1, 6)]
        else:
            k_list = kappa

        for level in k_list:
            list_aux = [eta_v[i] for i in range(level)]
            H.append(round(sum(list_aux), 4))

        return H

    def compute_Hs(self, op='true'):
        """Compute H for all vertices
        op 1: from true distribution
        op 2: from a priori distribution (which is the current estimated)
        op 3: from current estimated distribution
        """

        Hs, eta = [], []

        if op == 'true':
            eta = self.eta
        elif op == 'hat':
            eta = self.eta_hat
        else:
            exit()

        if self.m > 0:
            kappa = self.kappa
            for v in range(self.n):
                H_v = self.compute_H(eta[v], kappa)
                Hs.append(H_v)

            if op == 'true':
                self.H = Hs
            else:
                self.H_hat = Hs

        else:
            exit(print('Please provide thresholds'))

    # ------------------
    # immutable stuff
    # ------------------
    # UT - OK
    @staticmethod
    def define_danger_levels():
        """Danger level notation for all milp_sim"""
        danger_levels = [1, 2, 3, 4, 5]
        n_levels = len(danger_levels)
        level_label = ['Low', 'Moderate', 'High', 'Very High', 'Extreme']
        level_color = ['green', 'blue', 'yellow', 'orange', 'red']

        return danger_levels, n_levels, level_label, level_color

    # UT - ok
    @staticmethod
    def define_options():
        """Point estimate or pdf"""
        perception_list = ['point', 'prob', 'none']

        return perception_list

    @staticmethod
    def get_fov_ss2():
        """FOV is a dictionary
        fov[v] = [u1, u2...], vertices in line of sight of v"""
        fov = bf.fov_ss2()

        return fov

    @staticmethod
    def get_fov_9g():
        fov = dict()

        fov[1] = [(2, 1), (4, 1)]
        fov[2] = [(1, 1), (5, 1), (3, 1)]
        fov[3] = [(2, 1), (6, 1)]

        fov[4] = [(1, 1), (5, 1), (7, 1)]
        fov[5] = [(4, 1), (8, 1), (2, 1), (6, 1)]
        fov[6] = [(3, 1), (9, 1)]

        fov[7] = [(4, 1), (8, 1)]
        fov[8] = [(5, 1), (7, 1), (9, 1)]
        fov[9] = [(8, 1), (6, 1)]

        return fov

    # --------------------
    # handling data
    # --------------------
    @staticmethod
    def get_folder_path(sub_folder='danger_files'):
        """Return folder path to milp_sim > risk > scores folder"""

        folder_path = bf.get_folder_path('milp_sim', sub_folder, 'risk')

        return folder_path

    @staticmethod
    def load_danger_data(f_name, folder_path=None):
        """return danger data as dict data[v] = [eta1, ..., eta5]"""

        if folder_path is None:
            folder_path = MyDanger.get_folder_path()

        # danger_data[v] = [eta1, ..., eta5]
        danger_data = df.load_data(folder_path, f_name)

        return danger_data

    # ---------------------
    # scores
    # ---------------------
    # UT - OK
    @staticmethod
    def compute_frequentist(xi):
        """P(L=l) = sum_{I, C} xi/sum{L,I,C}
        xi[v] = [i1, i2,...], i1 = [l1, l2, l3, l4, l5]"""

        n_l = 5

        V = [v for v in xi.keys()]
        L = list(range(1, n_l + 1))

        # pt estimate
        z_hat = []
        # probabilities
        eta_hat = []

        for v in V:

            # scores for this node
            level_points = [0 for i in range(5)]
            all_points = 0.0

            # for each image data
            for img in xi[v]:

                # sum of all level points
                sum_all = 0

                for level in L:
                    l_idx = ext.get_python_idx(level)

                    # points for that level
                    xi_level = img[l_idx]

                    # for that vertex
                    level_points[l_idx] += xi_level
                    sum_all += xi_level

                all_points += sum_all

            # normalize for all points
            eta_v = [round(el * (1 / all_points), 4) for el in level_points]
            z_v = MyDanger.z_from_eta(eta_v)

            # estimates
            eta_hat.append(eta_v)
            # point estimate
            z_hat.append(z_v)

        return eta_hat, z_hat

    # UT - OK
    @staticmethod
    def load_scores(f_name='node_score_dict_Fire'):
        """Load pickle file with similarity scores"""
        # load file
        folder_path = MyDanger.get_folder_path()
        f_path = bf.assemble_file_path(folder_path, f_name, 'p')

        # xi_data[v] = [i1, i2, i3...], i = [xi_1, xi_2, xi_3..]
        xi_data = bf.load_pickle_file(f_path)

        return xi_data

    # ---------------------
    # not needed anymore (check and delete)
    # ---------------------

    # TODO get rid of this
    @staticmethod
    def not_sure_if_needed(f_name='node_danger_dict_01.p'):
        """Load pickle file with danger values"""
        # load file
        folder_path = bf.get_folder_path('milp_sim', 'scores', 'risk')

        # eta_data[v] = [eta_1, ..., eta_5]
        eta_data = df.load_data(folder_path, f_name)
        eta_true, z_true = [], []

        for v in eta_data.keys():
            eta_v = eta_data[v]
            eta_true.append(eta_v)

            # compute z_true
            z_v = MyDanger.z_from_eta(eta_v)
            z_true.append(z_v)

        return eta_true

    # TODO this may be redundant
    def set_danger_values(self, eta_true, eta_0):
        self.set_priori(eta_0)
        if isinstance(eta_true, str):
            self.gnd_truth_from_file(eta_true)
        else:
            self.gnd_truth_from_value(eta_true)

    def set_lookup(self, op=1):
        """Pre-Compute the estimated eta and z for all vertices, considering robot is at that vertex"""

        eta_hat, z_hat = [], []

        if op == 1:
            # if computing from matching scores
            xi = self.xi
            eta_hat, z_hat = self.compute_frequentist(xi)
        elif op == 2:
            # for another computation method
            eta_hat = self.eta
            z_hat = self.z
        else:
            exit()

        H_l = []
        for v in range(self.n):
            H_l.append(self.compute_H(eta_hat[v]))

        self.lookup_eta_hat = eta_hat
        self.lookup_z_hat = z_hat
        self.lookup_H_level = H_l









