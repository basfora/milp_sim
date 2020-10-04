from milp_sim.risk.classes.child_mespp import MySearcher2
from milp_mespp.core import extract_info as ext
import copy


class MyTeam2:
    """Dictionary with all searchers to facilitate things"""

    def __init__(self, m=1):
        """m: team size"""

        # --------------------------
        # ORIGINAL (backup) -- will not be changed
        # ---------------------------
        self.searchers_original = None
        # thresholds
        self.kappa_original = []
        self.alpha_original = []

        # --------------------------
        # ALIVE  -- updated and used during simulation
        # ---------------------------
        # size, members
        self.m = m
        self.S = ext.get_set_searchers(m)[0]
        # searcher objects (dictionary of searcher class)
        self.searchers = dict()
        # alive searchers -- original id (list)
        self.alive = self.S  # init as S (but will be different once an agent dies)
        # thresholds
        self.alpha = []
        self.kappa = []

        # ---------------------------
        # KILLED  -- updated and stored during simulation for future data analysis
        # ---------------------------
        # searchers killed (dict of searcher class)
        self.searchers_killed = dict()
        # killed searchers -- original id (list)
        self.killed = []
        # casualty info: dict[s_id_original] =  [vertex, time_step, z_true, kappa]
        self.killed_info = dict()
        # number of casualties
        self.casualties = 0

        # ------------------------------
        # retrieve info easily
        # ------------------------------
        # current position of searchers (dict)
        # TODO change to list later
        self.current_positions = dict()
        # start positions (list)
        self.start_positions = []

    def init_dict(self):

        for s_id in self.S:
            self.searchers[s_id] = None

    def update_size(self, m_or_S: int or list):
        S, m = ext.get_set_searchers(m_or_S)

        self.S, self.m = S, m

        self.alive = self.S

    # --------------------------
    # create searchers
    # --------------------------
    def add_searcher(self, s_id: int, v0: int, g, cap_s=0, zeta_s=None):
        """create each searcher"""

        self.searchers[s_id] = MySearcher2(s_id, v0, g, cap_s, zeta_s)

    def set_original(self):
        """Will not be changed"""
        self.searchers_original = copy.deepcopy(self.searchers)
        self.kappa_original = copy.deepcopy(self.kappa.copy())
        self.alpha_original = copy.deepcopy(self.alpha.copy())

    def create_dict_searchers(self, g, v0: list, kappa: list, alpha: list, capture_range=0, zeta=None):
        """Populate dictionary"""

        # set of searchers S = {1,..m}
        S, m = ext.get_set_searchers(v0)

        # create dict
        self.update_size(m)
        self.init_dict()

        if len(alpha) < 1:
            alpha = [0.95 for s_id in S]

        for s_id in S:
            v = ext.get_v0_s(v0, s_id)
            cap_s = ext.get_capture_range_s(capture_range, s_id)
            zeta_s = ext.get_zeta_s(zeta, s_id)

            # create each searcher
            self.add_searcher(s_id, v, g, cap_s, zeta_s)

        # set danger thresholds for each searcher
        self.set_thresholds(kappa, alpha)

        # set list for handy retrieve later
        self.set_start_positions()
        self.update_current_positions()

        # set original
        self.set_original()

        return self.searchers

    # -------------------------
    # set parameters (modify class)
    # -------------------------
    def set_thresholds(self, kappa: list, alpha: list):
        """Set danger threshold for all searchers"""
        self.kappa = kappa
        self.alpha = alpha

        for s_id in self.S:

            s_idx = ext.get_python_idx(s_id)
            k = kappa[s_idx]
            a = alpha[s_idx]

            s = self.searchers[s_id]
            s.set_kappa(k)
            s.set_alpha(a)

    def update_current_positions(self):
        """Update current positions of searchers
            """

        new_pos = dict()

        for s_id in self.S:
            s = self.searchers[s_id]
            # retrieve pos
            v = s.current_pos

            new_pos[s_id] = v

        self.current_positions = new_pos

        return self.current_positions

    def set_start_positions(self):
        start_pos = []

        for s_id in self.S:
            s = self.searchers[s_id]
            start_pos.append(s.start)

        self.start_positions = start_pos

        return self.start_positions

    def update_kappa(self):
        """Retrieve kappa from each searcher and return as list"""
        kappa_list = []

        for s_id in self.searchers.keys():
            s = self.searchers[s_id]
            kappa_list.append(s.kappa)

        self.kappa = kappa_list

        return self.kappa

    def update_alpha(self):
        """Retrieve alpha from each searcher and return as list"""
        alpha_list = []

        for s_id in self.searchers.keys():
            s = self.searchers[s_id]
            alpha_list.append(s.alpha)

        return self.alpha

    # --------------------------
    # danger related functions
    # --------------------------
    def get_H(self, danger):
        """list_H : list of cumulative danger probability,
        list_H = [H_v1, H_v2,...,H_vn], H_v1 = [H_s1, H_s2.. H_sm], H_s1 = sum_{l=1}^k eta_l, H_s1 in [0,1]"""

        list_H = []

        for v in range(danger.n):
            eta_v = danger.eta[v]
            H_v = []
            H_l = danger.compute_H(eta_v)
            for s_id in self.searchers.keys():
                kappa_idx = self.searchers[s_id].kappa - 1
                H_s = sum(H_l[:kappa_idx])
                H_v.append(H_s)
            list_H.append(H_v)

        return list_H

    def decide_searchers_luck(self, danger):
        killed = []

        # go through s in searchers (alive)
        for s_id in self.searchers.keys():
            # get searcher current id
            s = self.searchers[s_id]
            # retrieve current v
            v = s.current_pos
            # draw the searcher's luck
            casualty = danger.is_fatal(v)

            if casualty:
                # update life status
                s.set_alive(False)
                # retrieve original id
                id_0 = s.id_0
                # insert into searchers_killed
                self.searchers_killed[id_0] = s
                killed.append(id_0)

        # pop from searchers
        for el in killed:
            self.searchers.pop(el)

        # fix any changes in id
        self.update_searchers()

        return self.searchers, self.searchers_killed

    def update_searchers(self):
        """Fix id-ing of remaining searchers after killing spree"""

        new_searchers = dict()

        # fix the id of remaining searchers
        old_ids = [s_id for s_id in self.searchers.keys()]
        m = len(old_ids)

        new_id = 0
        for s_id in old_ids:
            new_id += 1
            s = self.searchers[s_id]
            # insert into new dict
            new_searchers[new_id] = s
            # update searchers id (do not touch the original id)
            new_searchers[new_id].set_new_id(new_id)

        self.searchers = new_searchers

        return new_searchers

    # -------------------------
    # retrieve information (do not modify class)
    # -------------------------
    def retrieve_current_positions(self, output_op=2):
        """Retrieve current positions of searchers
        Return:
            output_op = 1, list = [v_s1, ...v_sm]
            output_op = 2, dict_new_pos = {s: v}"""
        dict_new_pos = self.current_positions

        if output_op == 2:
            return dict_new_pos
        else:
            list_new_pos = []
            for s_id in self.S:
                list_new_pos.append(dict_new_pos[s_id])

            return list_new_pos

    def retrieve_start_positions(self):
        return self.start_positions

    def get_kappa(self):
        """Retrieve kappa from each searcher and return as list"""
        return self.kappa

    def get_alpha(self):
        """Retrieve alpha from each searcher and return as list"""
        return self.alpha


