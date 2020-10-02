"""All functions related to danger computation"""

from milp_mespp.core import data_fun as df, extract_info as ext
from milp_sim.risk.src import base_fun as bf


class MyDanger:
    """Define danger
    OBS: for ICRA simulations danger is constant for planning horizon"""

    def __init__(self, g, eta_true: list or int, eta0_0: list or int, plot_for_me=False):
        """
        :param g : graph
        :param xi : matching scores
        :param eta_true : ground truth danger level for each vertex
        :param eta0_0 : a priori probability of danger
        :param plot_for_me
         """

        # save graph
        self.g_name = ""
        self.g = None
        # name of folder for this sim: eg smoke_G9V_grid_date#_run#
        self.folder_name = ""
        # whole path + /name_folder
        self.whole_path = ""

        self.plot = plot_for_me

        # -----------
        # input parameters (immutable)
        # -----------
        # vertices
        self.V, self.n = [], 0
        # searchers
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

        # ---------------------
        # pre defined parameters
        # ---------------------
        # danger levels
        self.levels, self.n_levels, self.level_label, self.level_color = self.define_danger_levels()
        self.options = self.define_options()
        # ----------------------

        # get number of vertices
        self.structure(g)
        # set ground truth and a priori values
        self.set_danger_values(eta_true, eta0_0)
        # matching scores
        self.set_scores('node_score_dict_Fire.p')
        # pre compute estimated values (as if robot was in that vertex)
        self.set_lookup()
        # point or prob (default: point)
        self.perception = self.options[0]

    # called on init
    def structure(self, g):

        # graph info
        self.g_name = g["name"]
        self.g = g

        self.V, self.n = ext.get_set_vertices(g)

    def set_danger_values(self, eta_true, eta_0):
        self.set_priori(eta_0)
        self.set_true(eta_true)

    # UT - ok
    def set_priori(self, eta_0):

        n = self.n

        eta0_0, z0_0 = self.compute_apriori(n, eta_0)

        # eta_priori[v] = [eta_l1,... eta_l5]
        self.eta0_0 = eta0_0
        self.eta_hat = eta0_0

        H_levels = []
        for v in range(self.n):
            H_v = self.compute_H(eta0_0[v])
            H_levels.append(H_v)
        self.H_level_hat = H_levels

        # eta_priori_hat[v] = [level], level in {1,...5}
        self.z0_0 = z0_0
        self.z_hat = z0_0

        return

    def set_true(self, eta_true):

        if isinstance(eta_true, str):
            f_name = eta_true
            eta, z = self.load_gnd_truth(f_name)
        else:
            eta, z = self.compute_apriori(self.n, eta_true)

        # eta = [eta_l1,... eta_l5]
        self.eta = eta
        self.z = z

    def set_scores(self, xi: str or dict):

        if isinstance(xi, str):
            my_xi = MyDanger.load_scores(xi)
        else:
            my_xi = xi

        self.xi = my_xi

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

    def set_perception(self, option: int or str):
        """Set if point or probability estimate"""
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

    def save_estimated(self, z_hat, eta_hat, H_hat, t=0):
        self.stored_H_hat[t] = H_hat
        self.stored_z_hat[t] = z_hat
        self.stored_eta_hat[t] = eta_hat

    def vertex_lookup(self, v: int):

        v_idx = v - 1

        if self.perception == self.options[0]:
            # point
            my_hat = self.lookup_z_hat[v_idx]
        else:
            # probabilistic
            my_hat = self.lookup_eta_hat[v_idx]

        return my_hat

    def estimate(self):
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

    def compute_Hs(self, op=1):
        """Compute H for all vertices
        op 1: from true distribution
        op 2: from a priori distribution
        op 3: from current estimated distribution
        """

        Hs, eta = [], []

        if op == 1:
            eta = self.eta
        elif op == 2:
            eta = self.eta0_0
        elif op == 3:
            eta = self.eta_hat
        else:
            exit()

        if self.m > 0:
            kappa = self.kappa
            for v in range(self.n):
                H_v = self.compute_H(eta[v], kappa)
                Hs.append(H_v)

            if op == 1:
                self.H = Hs
            else:
                self.H_hat = Hs

        else:
            exit(print('Please provide thresholds'))

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
    def compute_apriori(n, my_eta=None):
        """Compute initial distribution
        :param n : vertices number |V|
        :param my_eta : int, list or None
        if no argument: set uniform probability
        """
        eta0_0, z0_0 = [], []

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
                    z0_0.append(MyDanger.z_from_eta(eta0_0[v]))
            # list of danger for each vertex
            else:
                z0_0 = my_eta
                for v in range(n):
                    eta0_0.append(MyDanger.eta_from_z(z0_0[v]))

        else:
            print('Wrong input for a priori danger values')
            exit()

        return eta0_0, z0_0

    @staticmethod
    def load_gnd_truth(f_name='node_danger_dict_01.p'):
        """Load pickle file with danger values"""
        # load file
        folder_path = bf.get_folder_path('milp_sim', 'scores', 'risk')

        # eta_data[v] = [eta_1, ..., eta_5]
        eta_data = df.load_data(folder_path, f_name)
        eta_true, z_true = [], []

        for v in eta_data.keys():
            eta_v = eta_data[v][-1]
            eta_true.append(eta_v)

            # compute z_true
            z_v = MyDanger.z_from_eta(eta_v)
            z_true.append(z_v)

        return eta_true

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
    def z_from_eta(eta_v: list, k=None, op=1):
        """
        op 1: conservative (max value)
        op 2: # min threshold for team """
        z = 1

        if op == 1:
            # conservative (max value)
            z = MyDanger.max_z_from_eta(eta_v)
        elif op == 2:
            # min threshold for team
            z = MyDanger.break_z_tie(eta_v, k)
        else:
            exit(print('No other option!'))

        return z

    # UT - ok
    @staticmethod
    def max_z_from_eta(eta_v: list):
        """eta_hat = max prob level (conservative!)
        eta = [eta_l1,... eta_l5]
        return last occurrence (max danger value)
        return integer"""

        z = max(idx + 1 for idx, val in enumerate(eta_v) if val == max(eta_v))

        return z

    # UT - ok
    @staticmethod
    def break_z_tie(eta_v: list, kappa: list):
        """eta_hat = max prob level
        eta = [eta_l1,... eta_l5]
        return last occurrence (max danger value)
        return integer"""

        z_list = [idx + 1 for idx, val in enumerate(eta_v) if val == max(eta_v)]

        # min team threshold
        k_min = min(kappa)

        z = 1
        for my_z in z_list:
            if my_z > k_min:
                break
            z = my_z

        return z

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

    # UT - OK
    @staticmethod
    def load_scores(f_name='node_score_dict_Fire.p'):
        """Load pickle file with similarity scores"""
        # load file
        folder_path = bf.get_folder_path('milp_sim', 'scores', 'risk')

        # xi_data[v] = [i1, i2, i3...], i = [xi_1, xi_2, xi_3..]
        xi_data = df.load_data(folder_path, f_name)

        return xi_data

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


