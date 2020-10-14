"""All functions related to danger computation"""
import random
from milp_mespp.core import extract_info as ext
from milp_sim.risk.src import base_fun as bf
import copy
import math


class MyDanger:
    """Define danger
    OBS: for ICRA simulations danger is constant for planning horizon"""

    def __init__(self, g):
        """
        :param g : graph
         """

        # ---------------------
        # pre defined parameters
        # ---------------------
        # danger levels
        self.levels, self.n_levels, self.level_label, self.level_color = self.define_danger_levels()
        self.options = self.define_options()

        # ------------------------
        # Default settings
        # ------------------------
        # apply danger constraints
        self.constraints = True
        # apply kill probability
        self.kill = True
        # true knowledge
        self.true_priori = False
        self.uniform_priori = True
        self.true_estimate = False
        self.mva_conservative = True
        # z true always gonna be mean, for now estimate also going to be mean (so it doesn't get stuck)
        self.z_true_op = 1
        self.z_est_op = 1
        # z for priori is gonna be conservative (op = 4)
        self.z_pri_op = 4
        self.use_fov = True
        # point or prob (default: point)
        self.perception = self.options[0]

        # -----------
        # input parameters (graph, immutable)
        # -----------
        # vertices
        self.V, self.n = ext.get_set_vertices(g)
        self.fov = None

        # ----------
        # team of searchers (get rid of this to avoid issues when searcher is dead?)
        # ----------
        self.S, self.m = [], 0
        # danger threshold for the searchers
        self.kappa = list()
        self.alpha = list()
        self.k_mva = None
        self.v0 = [1]

        # -------------------------
        # ground truth
        # -------------------------
        # eta = [eta_l1, ... eta_l5]
        self.eta = []
        # z = argmax eta_l \in {1,..5}
        self.z = []
        # probabilistic approach H = [cum for each level]
        self.H = []

        # probability kill
        self.prob_kill = None

        # ------------
        # estimation
        # ------------

        # a priori knowledge of danger
        # eta0_0[v] = [eta0_0_l1,... eta0_0_l5]
        self.eta0_0 = []
        # z0_0[v] = level, level in {1,...5}
        self.z0_0 = []
        self.H0_0 = []

        # estimates at each time step
        # eta_hat[v] = [l1,...l5]
        self.eta_hat = []
        # probable danger level
        self.z_hat = []

        # for each searcher
        # H[s] = [H_s1, H_s2...H_sm] (estimate)
        self.H_hat = []

        # to make getting the values easier
        # look up of estimated danger for each vertex (when robots see every image in that vertex)
        self.lookup_eta_hat = []
        # H_l[v] = [l1...l5] (true value)
        self.lookup_H_hat = []
        self.lookup_z_hat = []
        # matching scores xi[v] = [[i1_1...i1_5], [i2_1,...i2_5]]
        self.xi = dict()

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
        f_name = 'exp_data'
        if self.perception == 'prob':
            f_name = 'exp_data_prob'
        self.path_exp_data = self.get_folder_path(f_name)

        # -----------------------
        # danger data from files
        # -----------------------
        # true distributions (100% images)
        self.true_file_name = ''
        self.true_file_path = ''
        self.true_raw = None
        # estimated offline (% images)
        self.estimated_file_name = ''
        self.estimated_file_path = ''
        self.percentage_im = None
        self.estimated_raw = None
        self.extension = 'pkl'
        # path to files
        self.folder_path = self.get_folder_path()
        # ----------------------

    # ------------------------------------------------
    # experimental setups
    # ------------------------------------------------
    def set_true_priori(self, status=False):
        self.true_priori = status
        self.true_estimate = status

    def set_mva_conservative(self, status=True):
        self.mva_conservative = status
        self.set_z_comp()

    def set_z_comp(self, op=1):
        """Choose how to compute z:
        op 1: get maximum prob, if tie pick the one closest to the middle
        op 2: get maximum prob, if tie pick conservative (max value)
        op 3: get max prob, if tie pick min threshold for team (break ties)
        op 4: get max prob, if tie pick min threshold + 1"""

        if self.mva_conservative:
            self.z_pri_op = 4
            if len(self.kappa) > 0:
                self.k_mva = min(self.kappa)
        else:
            self.z_pri_op = op

    def set_true_estimate(self, status=False):
        self.true_estimate = status

    def set_constraints(self, status=True):
        self.constraints = status

    def set_kill(self, status=True, op=3):
        self.kill = status

        if status:
            if isinstance(op, int):
                prob_kill = self.exp_prob_kill(op)
            else:
                prob_kill = op
        else:
            prob_kill = None

        self.prob_kill = prob_kill

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

    def set_use_fov(self, status=True, op_test=False):
        self.use_fov = status

        if self.use_fov:
            if op_test:
                self.fov = self.get_fov_9g()
            else:
                self.fov = self.get_fov_ss2()
        else:
            self.fov = None

    # ----------------------
    # Define true, priori and estimate
    # ----------------------

    def set_true(self, eta_true):
        """Set up true danger values, input required
        :param eta_true : str of file or value"""

        # ground truth (needs to input)
        if isinstance(eta_true, str):
            self.gnd_truth_from_file(eta_true, self.extension)
        else:
            # useful for unit tests and sanity checks
            self.gnd_truth_from_value(eta_true)

        if self.true_priori:
            self.set_priori()

        if self.true_estimate:
            self.set_estimate()

    def set_priori(self, eta_priori=None):
        n = self.n

        # use true value for priori if argument is none and true_priori is True
        if self.true_priori:
            eta0_0, z0_0, H0_0 = self.copy_true_value()
        elif self.uniform_priori:
            eta0_0, z0_0 = self.compute_uniform(self.n, self.z_pri_op, self.k_mva)
            H0_0 = self.compute_all_H(eta0_0)
        else:
            # input probability
            eta0_0, z0_0 = self.compute_from_value(n, eta_priori, self.z_pri_op, self.k_mva)
            H0_0 = self.compute_all_H(eta0_0)

        eta0_0, z0_0, H0_0 = self.set_v0_danger(eta0_0, z0_0, H0_0)

        # eta_priori[v] = [eta_l1,... eta_l5]
        self.eta0_0 = eta0_0
        # z_priori[v] = [level], level in {1,...5}
        self.z0_0 = z0_0
        self.H0_0 = H0_0

        # first estimate is a priori
        self.eta_hat = eta0_0
        self.z_hat = z0_0
        self.H_hat = H0_0

        # save in storage
        self.stored_eta_hat[0] = eta0_0
        self.stored_z_hat[0] = z0_0
        self.stored_H_hat[0] = H0_0

        return

    def set_estimate(self, etahat_off=None):

        # estimated danger from file
        if isinstance(etahat_off, str):
            per = int(etahat_off.split('_')[-1])
            percentage = per
            self.estimated_from_file(etahat_off, percentage, self.extension)
        else:
            # overwrite if it's supposed to be true estimate
            if self.true_estimate:
                etahat_off = None

            # useful for unit tests and sanity checks
            self.estimated_from_value(etahat_off)

    # ----------------------
    # From files --- load danger data and get ready for simulation
    # ----------------------

    def gnd_truth_from_file(self, f_true: str, extension='pkl'):
        """Danger data (true): set files, load and format to simulation"""

        # point to file, which should be in danger_files folder
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
            self.z = self.all_z_from_eta(self.eta, self.z_true_op,  None)
            self.H = self.compute_all_H(self.eta)

            self.eta, self.z, self.H = self.set_v0_danger(self.eta, self.z, self.H)

        elif op == 'hat':

            eta_hat = bf.load_pickle_file(self.estimated_file_path)
            self.estimated_raw = eta_hat
            # set lookup table
            etahat_list = bf.is_list(self.estimated_raw)
            # make it easier, lookup eta_hat
            self.lookup_eta_hat = etahat_list
            # compute z_hat
            self.lookup_z_hat = self.all_z_from_eta(self.lookup_eta_hat, self.z_est_op, self.k_mva)
            # compute H_hat
            self.lookup_H_hat = self.compute_all_H(self.lookup_eta_hat)

        return

    def set_v0_danger(self, eta, z, H):

        my_eta = [1, 0, 0, 0, 0]
        my_z = 1
        my_H = [1, 0, 0, 0, 0]

        for v in self.v0:
            vidx = ext.get_python_idx(v)
            eta[vidx] = my_eta
            z[vidx] = my_z
            H[vidx] = my_H

        return eta, z, H

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

    # UT - ok
    def gnd_truth_from_value(self, eta_true):

        eta, z = self.compute_from_value(self.n, eta_true, self.z_true_op, self.k_mva)

        # eta = [eta_l1,... eta_l5]
        self.eta = eta
        self.z = z

    def estimated_from_value(self, my_eta_hat):
        """Mostly for testing purposes"""

        z_hat = None
        H_hat = None

        if my_eta_hat is None:
            # make it equal to gnd truth
            eta_hat, z_hat, H_hat = self.copy_true_value()
        elif my_eta_hat == 0:
            eta_hat = copy.copy(self.eta0_0)
            z_hat = copy.copy(self.z0_0)
            H_hat = copy.copy(self.H0_0)
        elif my_eta_hat == 1:
            # frequentist (just for testing), computing from matching scores
            xi = self.xi
            eta_hat, z_hat = self.compute_frequentist(xi)
            H_hat = self.compute_all_H(eta_hat)
        else:
            eta_hat = bf.is_list(my_eta_hat)

        self.set_lookup_hat(eta_hat, z_hat, H_hat)

    def set_lookup_hat(self, eta_hat, z_hat=None, H_hat=None):
        """Pre-Compute the estimated eta and z for all vertices,
         considering robot is at that vertex"""

        # distribution
        self.lookup_eta_hat = eta_hat

        # point estimate
        if z_hat is None:
            # compute z
            self.lookup_z_hat = self.all_z_from_eta(self.lookup_eta_hat, self.z_est_op, self.k_mva)
        else:
            self.lookup_z_hat = z_hat

        # cumulative distribution for each level
        self.lookup_H_hat = H_hat

    def set_scores(self, xi: str or dict):

        if isinstance(xi, str):
            my_xi = MyDanger.load_scores(xi)
        else:
            my_xi = xi

        self.xi = my_xi

    # -----------------
    # set or update parameters
    # -----------------
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
                k = [kappa for s in self.S]
                a = [alpha for s in self.S]

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
            try:
                t = ext.get_last_key(self.stored_z_hat)
            except:
                t = 0

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

    # UT - ok
    def estimate(self, visited_vertices):
        """Estimate danger level based on:
        current estimate + current position (new measurements)
        FOV of current vertices
        a priori (for non-visited/non-line of sight vertices)
        """

        if self.true_priori:
            eta_hat, z_hat, H_hat = self.perfect_estimate()
        elif self.use_fov is False:
            eta_hat, z_hat, H_hat = self.no_fov_estimate(visited_vertices)
        else:
            eta_hat, z_hat, H_hat = self.normal_estimate(visited_vertices)

        self.update_estimate(eta_hat, z_hat, H_hat)
        self.save_estimate()

    def normal_estimate(self, visited_vertices: list):
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
        H_hat = []

        # loop through vertices
        for v in self.V:
            # get priori (if no info available, it will keep it)
            etahat_v = self.get_etahat(v)
            zhat_v = self.get_zhat(v)
            Hhat_v = self.get_Hhat(v)

            # if vertex was visited, get estimated from lookup table
            if v in visited_vertices:
                etahat_v, zhat_v, Hhat_v = self.get_from_lookup(v)

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
                            etahat_v, zhat_v, Hhat_v = self.get_from_lookup(u)
                            break

            # append to estimated list
            eta_hat.append(etahat_v)
            z_hat.append(zhat_v)
            H_hat.append(Hhat_v)

        return eta_hat, z_hat, H_hat

    def perfect_estimate(self):
        """Estimate will always be the perfect danger value"""

        # new update
        eta_hat = copy.copy(self.eta)
        z_hat = copy.copy(self.z)
        H_hat = copy.copy(self.H)

        return eta_hat, z_hat, H_hat

    def no_fov_estimate(self, visited_vertices: list):
        """Update danger values only of visited vertices"""

        # new update
        eta_hat = []
        z_hat = []
        H_hat = []

        # loop through vertices
        for v in self.V:
            # get priori (if no info available, it will keep it)
            etahat_v, zhat_v, Hhat_v = self.get_priori(v)

            # if vertex was visited, get estimated from lookup table
            if v in visited_vertices:
                etahat_v, zhat_v, Hhat_v = self.get_from_lookup(v)

            # append to estimated list
            eta_hat.append(etahat_v)
            z_hat.append(zhat_v)
            H_hat.append(Hhat_v)

        return eta_hat, z_hat, H_hat

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
            prob_kill = self.exp_prob_kill()
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
    def exp_prob_kill(op=1):
        levels = MyDanger.get_levels()
        prob_kill = []
        if op == 1:
            a = 0.0035
            prob_kill = [a * math.exp(level) for level in levels]
        elif op == 2:
            a = 3.3e-5
            prob_kill = [a * level * math.exp(level) for level in levels]
        elif op == 3:
            a = 0.00009
            b = 1/1200
            prob_kill = [a + (level-1) * b * math.exp(level) for level in levels]
        else:
            exit()

        return prob_kill

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
    def copy_true_value(self):
        eta = copy.deepcopy(self.eta)
        z = copy.deepcopy(self.z)
        H = copy.deepcopy(self.H)

        return eta, z, H

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

    def get_priori(self, v: int):
        v_idx = ext.get_python_idx(v)

        z0 = self.z0_0[v_idx]
        eta0 = self.eta0_0[v_idx]
        H0_0 = self.H0_0[v_idx]

        return eta0, z0, H0_0

    def get_H(self, v: int):
        v_idx = ext.get_python_idx(v)
        true_H = self.H[v_idx]
        return true_H

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

    def get_Hhat(self, v: int):
        # get level for that vertex (z_hat)
        v_idx = ext.get_python_idx(v)
        H_hat = self.H_hat[v_idx]

        return H_hat

    def get_from_lookup(self, v: int):
        # get level for that vertex (eta_hat)
        v_idx = ext.get_python_idx(v)
        eta_hat = self.lookup_eta_hat[v_idx]
        z_hat = self.lookup_z_hat[v_idx]
        H_hat = self.lookup_H_hat[v_idx]

        return eta_hat, z_hat, H_hat

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

    def get_true(self, v: int):
        """Return true danger values for that vertex"""

        v_idx = ext.get_python_idx(v)
        eta_hat = self.lookup_eta_hat[v_idx]
        z_hat = self.lookup_z_hat[v_idx]

        return eta_hat, z_hat

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
    def compute_from_value(n, my_eta, op, k=None):
        """Compute initial distribution
        :param n : vertices number |V|
        :param my_eta : int, list or None
        :param op
        :param k
        if eta=None argument: set uniform probability
        """
        eta0_0, z0_0 = [], []

        if my_eta is None:
            print('Setting uniform probability')
            for v in range(n):
                eta0_v = [0.2 for i in range(5)]
                z_v = MyDanger.z_from_eta(eta0_v, op, k)

                eta0_0.append(eta0_v)
                z0_0.append(z_v)

        elif isinstance(my_eta, int):
            for v in range(n):
                eta0_0.append(MyDanger.eta_from_z(my_eta))
                z0_0.append(my_eta)

        elif isinstance(my_eta, list):
            # list of lists (prob for each vertex)
            if isinstance(my_eta[0], list):
                eta0_0 = my_eta
                for v in range(n):
                    z0_0.append(MyDanger.z_from_eta(eta0_0[v], op, k))
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
    def compute_uniform(n, op, k=None):
        eta0_0, z0_0 = [], []

        print('Setting uniform probability')
        for v in range(n):
            eta0_v = [0.2 for i in range(5)]
            z_v = MyDanger.z_from_eta(eta0_v, op, k)

            eta0_0.append(eta0_v)
            z0_0.append(z_v)

        return eta0_0, z0_0

    # --------------------
    # distribution to point estimate and vice versa
    # --------------------

    @staticmethod
    def all_z_from_eta(eta: list or dict, op, k=None):
        """Return list of point estimate for all vertices"""

        eta = bf.is_list(eta)

        n = len(eta)
        z = []

        for vidx in range(n):
            eta_v = eta[vidx]
            z_v = MyDanger.z_from_eta(eta_v, op, k)
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
    def z_from_eta(eta_v: list, op, k=None):
        """
        op 1: get maximum prob, if tie pick the one closest to the middle
        op 2: get maximum prob, if tie pick conservative (max value)
        op 3: get max prob, if tie pick min threshold for team (break ties)
        op 4: get max prob, if tie pick min threshold + 1"""

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
        elif op == 4:
            # get max prob, if tie pick min threshold + 1
            z = MyDanger.pick_z_for_mva(z_list, k)
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
    def pick_z_from_kappa(z_list: list, kappa: list or int):
        """eta_hat = max prob level
        eta = [eta_l1,... eta_l5]
        return last occurrence (max danger value)
        return integer"""

        # min team threshold
        if isinstance(kappa, list):
            k_min = min(kappa)
        else:
            k_min = kappa

        z = 1
        for my_z in z_list:
            if my_z > k_min:
                break
            z = my_z

        return z

        # UT - ok

    @staticmethod
    def pick_z_for_mva(z_list: list, kappa: list or int):
        """eta_hat = max prob level
        eta = [eta_l1,... eta_l5]
        return last occurrence (max danger value)
        return integer"""

        # min team threshold
        if isinstance(kappa, list):
            k_min = min(kappa)
        else:
            k_min = kappa

        z = 1
        for my_z in z_list:
            z = my_z
            if my_z > k_min:
                break

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

        levels = [1, 2, 3, 4, 5]

        if kappa is None:
            k_list = [level for level in levels]
        else:
            k_list = kappa

        for level in k_list:
            list_aux = [eta_v[i] for i in range(level)]
            H.append(round(sum(list_aux), 3))

        H[-1] = 1.0

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

    @staticmethod
    def get_levels():
        danger_levels, n_levels, level_label, level_color = MyDanger.define_danger_levels()

        return danger_levels

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

        f_path = bf.assemble_file_path(folder_path, f_name, 'pkl')

        # danger_data[v] = [eta1, ..., eta5]
        danger_data = bf.load_pickle_file(f_path)

        return danger_data

    # ---------------------
    # scores
    # ---------------------
    # UT - OK
    @staticmethod
    def compute_frequentist(xi: dict, op=1):
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
            z_v = MyDanger.z_from_eta(eta_v, op)

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

    # TODO this may be redundant
    def set_danger_values(self, eta_true, eta_0):
        self.set_priori(eta_0)
        if isinstance(eta_true, str):
            self.gnd_truth_from_file(eta_true)
        else:
            self.gnd_truth_from_value(eta_true)

    @staticmethod
    def fake_data_for_testing():

        # get scores
        xi = MyDanger.load_scores()

        # true danger level
        eta, z = MyDanger.compute_frequentist(xi)

        # collect part of the scores only
        xi_25 = dict()
        xi_10 = dict()
        for node in xi.keys():
            n_img = len(xi[node])
            n_25 = math.ceil((1/4) * n_img)
            n_10 = math.ceil((1/10) * n_img)

            scores = [xi[node][i] for i in range(0, n_25)]
            scores_10 = [xi[node][i] for i in range(0, n_10)]
            xi_25[node] = scores
            xi_10[node] = scores_10

        # estimated danger level
        eta_hat, z_hat = MyDanger.compute_frequentist(xi_25)
        eta_10, z_10 = MyDanger.compute_frequentist(xi_10)

        # fake data
        data_25 = bf.list_to_dict(eta_hat)
        data_100 = bf.list_to_dict(eta)
        data_10 = bf.list_to_dict(eta_10)

        # save pickle file
        base_path = MyDanger.get_folder_path()
        extension = 'pkl'
        # file name
        f_25 = 'fake_01_25'
        f_100 = 'fake_01_100'
        f_10 = 'fake_01_10'
        # full path
        path_100 = bf.assemble_file_path(base_path, f_100, extension)
        path_25 = bf.assemble_file_path(base_path, f_25, extension)
        path_10 = bf.assemble_file_path(base_path, f_10, extension)
        # save
        bf.make_pickle_file(data_100, path_100)
        bf.make_pickle_file(data_25, path_25)
        bf.make_pickle_file(data_10, path_10)

    @staticmethod
    def print_danger_data():
        f_name = 'danger_map_100'
        f_name25 = 'danger_map_25'
        f_name50 = 'danger_map_50'
        f_name75 = 'danger_map_75'

        data = MyDanger.load_danger_data(f_name)
        data_25 = MyDanger.load_danger_data(f_name25)
        data_50 = MyDanger.load_danger_data(f_name50)
        data_75 = MyDanger.load_danger_data(f_name75)

        op = 1

        for v in data.keys():
            eta_true = data.get(v)
            eta_25 = data_25.get(v)
            eta_50 = data_50.get(v)
            eta_75 = data_75.get(v)

            z_true = MyDanger.z_from_eta(eta_true, op)
            z_25 = MyDanger.z_from_eta(eta_25, op)
            z_50 = MyDanger.z_from_eta(eta_50, op)
            z_75 = MyDanger.z_from_eta(eta_75, op)

            print('Vertex %d:\nDistribution:  eta_true = %s, eta_25 = %s, eta_50 = %s, eta_75 = %s'
                  % (v, str(eta_true), str(eta_25), str(eta_50), str(eta_75)))
            print('Point Estimate, z_true = %d, z_25 = %d, z_50 = %d, z_75 = %d] \n ---'
                  % (z_true, z_25, z_50, z_75))

    @staticmethod
    def print_fake_data():
        f_name = 'fake_01_100'
        f_name10 = 'fake_01_10'
        f_name25 = 'fake_01_25'

        data = MyDanger.load_danger_data(f_name)
        data_25 = MyDanger.load_danger_data(f_name25)
        data_10 = MyDanger.load_danger_data(f_name10)

        op = 1

        for v in data.keys():
            eta_true = data.get(v)
            eta_25 = data_25.get(v)
            eta_10 = data_10.get(v)

            z_true = MyDanger.z_from_eta(eta_true, op)
            z_25 = MyDanger.z_from_eta(eta_25, op)
            z_10 = MyDanger.z_from_eta(eta_10, op)

            print('Vertex %d:\nDistribution:  eta_true = %s, eta_25 = %s, eta_10 = %s'
                  % (v, str(eta_true), str(eta_25), str(eta_10)))
            print('Point Estimate, z_true = %d, z_25 = %d, z_10 = %d \n ---' % (z_true, z_25, z_10))







