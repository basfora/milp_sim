from milp_sim.risk.classes.child_mespp import MySearcher2
from milp_sim.risk.src import base_fun as bf
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
        self.perception = 'point'
        self.size = m

        # --------------------------
        # ALIVE  -- updated and used during simulation
        # ---------------------------
        # current team size and members (lexicographic order - not original id)
        self.m = m
        self.S = ext.get_set_searchers(m)[0]
        # searcher objects (dictionary of searcher class)
        self.searchers = dict()
        # alive searchers -- original id (list)
        self.alive = self.S  # init as S (but will be different once an agent dies)
        # thresholds
        self.alpha = []
        self.kappa = []
        self.hh = False
        self.mva = None

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
        # vertices visited (for danger estimation FOV)
        self.visited_vertices = []

    def init_dict(self):

        for s_id in self.S:
            self.searchers[s_id] = None

    # --------------------------
    # create searchers
    # --------------------------
    def add_searcher(self, s_id: int, v0: int, g, cap_s=0, zeta_s=None):
        """create each searcher"""

        self.searchers[s_id] = MySearcher2(s_id, v0, g, cap_s, zeta_s)

    def set_original(self):
        """Will not be changed"""
        self.searchers_original = copy.deepcopy(self.searchers)
        self.kappa_original = copy.deepcopy(self.kappa)
        self.alpha_original = copy.deepcopy(self.alpha)

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
        self.update_pos_list()

        # set original
        self.set_original()
        self.update_alive()

        return self.searchers

    def set_danger_perception(self, op='point'):

        for s_id in self.searchers_original:
            s = self.searchers_original[s_id]
            s.set_perception(op)

        for s_id in self.searchers:
            s = self.searchers[s_id]
            s.set_perception(op)

        self.perception = op

    # -------------------------
    # set and update parameters (modify class)
    # -------------------------
    def set_thresholds(self, kappa: list, alpha: list):
        """Set danger threshold for all searchers"""
        self.kappa = kappa
        self.alpha = alpha

        self.is_homogeneous()

        for s_id in self.S:

            s_idx = ext.get_python_idx(s_id)
            k = kappa[s_idx]
            a = alpha[s_idx]

            s = self.searchers[s_id]
            s.set_kappa(k)
            s.set_alpha(a)

        if self.hh is False:
            self.searchers[self.mva].set_mva()

    def is_homogeneous(self):
        unique = []
        for k in self.kappa:
            if k not in unique:
                unique.append(k)

        if len(unique) == 1:
            self.hh = True
        else:
            self.hh = False
            min_k = min(self.kappa)
            self.mva = self.kappa.index(min_k) + 1

    def set_start_positions(self):
        start_pos = []

        for s_id in self.S:
            s = self.searchers[s_id]
            start_pos.append(s.start)

        self.start_positions = start_pos
        self.update_pos_list()
        self.update_vertices_visited()

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

        self.alpha = alpha_list

        return self.alpha

    def update_alive(self):
        """Check for searchers alive, update list with original ids"""

        alive = []

        for s_alive in self.searchers.keys():
            s = self.searchers.get(s_alive)

            # sanity check
            if s.alive:
                alive.append(s.id_0)
            else:
                exit(print('Something is wrong, searchers not updated.'))

        self.alive = alive

        if self.S != self.alive:
            # update current team size
            m = len(self.alive)
            self.update_size(m)

    def update_size(self, m_or_S: int or list or None):

        if m_or_S is None:
            m_or_S = len(self.alive)

        self.S, self.m = ext.get_set_searchers(m_or_S)

    # --------------------------
    # moving functions
    # --------------------------
    def searchers_evolve(self, new_pos):
        """call to evolve searchers position
        new_pos = {s: v}"""

        for s_id in self.searchers.keys():
            self.searchers[s_id].evolve_position(new_pos[s_id])

        self.update_pos_list()
        self.update_vertices_visited()

        return self.searchers

    def update_pos_list(self):
        """Update current positions of searchers"""

        new_pos = dict()

        for s_id in self.S:
            v = self.searchers[s_id].current_pos
            new_pos[s_id] = v

        self.current_positions = new_pos

        return self.current_positions

    def update_vertices_visited(self):
        """list of vertices visited by the searchers"""

        for s_id in self.searchers.keys():
            s = self.searchers.get(s_id)
            # retrieve last vertex
            t, v = ext.get_last_info(s.path_taken)

            if v not in self.visited_vertices:
                self.visited_vertices.append(v)

        return self.visited_vertices

    # UT - OK
    def get_path(self):
        """Return path[s, t] = v"""

        path = {}
        for s_id in self.searchers.keys():
            list_v = self.searchers[s_id].get_last_planned()

            t = 0
            for v in list_v:
                path[(s_id, t)] = v
                t += 1

        return path

    # UT - OK
    def get_path_list(self):
        """Get path, return as list for each searcher s
            path[s] = [v0, v1, v2...vh]"""

        path_dict = self.get_path()
        path_list = ext.path_as_list(path_dict)

        return path_list

    def print_summary(self):

        print('--\n--')

        for s_id in self.searchers.keys():
            s = self.searchers.get(s_id)
            id_0 = s.id_0
            path_taken = bf.dict_to_list(s.path_taken)
            print('Searcher %d visited: %s ' % (id_0, str(path_taken)))

        for id_0 in self.killed_info.keys():
            s = self.searchers_killed[id_0]
            path_taken = bf.dict_to_list(s.path_taken)
            info = self.killed_info.get(id_0)
            print('Searcher %d visited: %s ' % (id_0, str(path_taken)))
            print('Searcher %d was killed at t = %d, vertex %d (danger level %d)' % (id_0, info[1], info[0], info[2]))

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

    def to_kill_or_not_to_kill(self, danger, t: int):
        """Decide which robots will die based on their
        current vertex, true danger level of that vertex and probability of casualty
        default: [0.1, 0.2, 0.3, 0.4, 0.5]"""

        # death note list
        killed_ids = []

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

                # save in searchers killed (dict[original id])
                self.searchers_killed[id_0] = copy.deepcopy(s)
                # save in killed list (original id)
                self.killed.append(id_0)
                # increase number of casualties
                self.casualties += 1

                # killing info: vertex, time-step, true danger value, threshold
                z = danger.get_z(v)
                self.killed_info[id_0] = [v, t, z, s.kappa]

                print('Oh no! Searcher %d was killed on vertex %d (danger level %d)' % (id_0, v, z))

                # insert into killed_ids list (original id)
                killed_ids.append(id_0)

        return killed_ids

    def decide_searchers_luck(self, danger, t=0):
        """Will the robots survive the danger?
        To kill or not to kill, that is the question
        Kill and update searchers, searchers_killed, ids, team size and alive list
        """

        # update self.killed list
        self.to_kill_or_not_to_kill(danger, t)

        # fix any changes in id based on self.killed
        self.update_searchers_ids()
        self.update_size(len(self.searchers))

        # update list of alive searchers (original id) based on self.searchers
        self.update_alive()

        # update current positions
        self.update_pos_list()

        # update list of team's thresholds
        self.update_alpha()
        self.update_kappa()

        return self.searchers, self.searchers_killed

    def update_searchers_ids(self):
        """Fix id-ing of remaining searchers after killing spree"""

        current_ids = [s_id for s_id in self.searchers.keys()]

        # pop from searchers based on killed list
        if len(self.killed) > 0:
            # not very efficient but sanity check
            for s_id in current_ids:
                id0 = self.searchers[s_id].id_0
                if id0 in self.killed:
                    self.searchers.pop(s_id)

        # get the current id of searchers alive
        old_ids = [s_alive for s_alive in self.searchers.keys()]
        m = len(old_ids)

        # create new dict
        new_searchers = dict()
        if m > 0:

            new_id = 0
            for s_alive in old_ids:
                # do a lexicographic order
                new_id += 1
                # get searcher that is still alive
                s = self.searchers.get(s_alive)
                # insert into new dict
                s_copy = copy.deepcopy(s)
                new_searchers[new_id] = s_copy
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
            output_op = 2, dict_new_pos = {s: v} <-- path_next_t"""
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


