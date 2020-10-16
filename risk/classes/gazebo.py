"""Class for Gazebo Sims"""
from milp_sim.risk.classes.child_mespp import MyInputs2
from milp_sim.risk.src import plan_risk as plnr
from milp_mespp.core import plan_fun as pln
from milp_mespp.core import sim_fun as sf, extract_info as ext
import copy


class MyGazeboSim:

    def __init__(self, v0: list, b_0: list, visited: list, t: int, sim_op=1):

        # original parameters
        self.kappa_original = [3, 4, 5]
        self.alpha_original = [0.95, 0.95, 0.95]
        self.m_original = 3
        self.S_original = [1, 2, 3]
        self.id_map = []

        self.kappa = copy.copy(self.kappa_original)
        self.alpha = copy.copy(self.alpha_original)

        # self.id = 0
        self.time_step = t
        self.sim_op = sim_op
        # status prior to this iteration
        self.alive, self.killed, v_current = self.adjust_team(v0)
        self.adjust_threshold()

        # for this iteration
        self.m_alive = len(self.alive)

        # plan[s] = [v0, .... vh]
        self.plan = {}
        # belief_vector = [b_c, b_1,....b_n]
        self.belief_vector = []
        # vertices visited = [v, u, w,..]
        self.visited = []
        self.v0 = [1]

        # here only with alive
        self.call_planner(v_current, b_0, visited)

    # --------------------------------------------------------------------------------
    # Adjusting team sizes
    # --------------------------------------------------------------------------------
    def adjust_team(self, v_current: list):
        """ OK"""
        alive = []
        killed = []
        v0 = []

        i = 0
        for s_id in self.S_original:
            v = v_current[s_id - 1]
            # if it already died, don't count with it
            if v < 0:
                killed.append(s_id)
                self.id_map.append((s_id, -1))
            else:
                alive.append(s_id)
                v0.append(v)
                s_new = len(v0)
                self.id_map.append((s_id, s_new))

        return alive, killed, v0

    def adjust_threshold(self):
        """"OK"""
        kappa = []
        alpha = []

        for s_id in self.S_original:
            if self.is_alive(s_id, self.alive):
                s_idx = s_id - 1
                kappa.append(self.kappa_original[s_idx])
                alpha.append(self.alpha_original[s_idx])

        self.kappa = kappa
        self.alpha = alpha

    @staticmethod
    def is_alive(s_id, alive):
        if s_id in alive:
            return True
        else:
            return False

    def set_path_to_output(self, just_killed: list, path=None):

        if path is None or len(just_killed) == self.m_original:
            return None

        else:
            path_list = ext.path_as_list(path)

            for s_id in self.S_original:
                # get new id
                new_id = self.get_new_id(s_id)

                if new_id < 0 or new_id in just_killed:
                    # was killed before this iteration
                    self.plan[s_id] = self.make_dummy_path()
                else:
                    self.plan[s_id] = path_list[new_id]
        return

    def get_new_id(self, s_id):

        for couple in self.id_map:
            if couple[0] == s_id:
                return couple[1]

    def make_dummy_path(self):
        return  [-1 for i in range(self.specs.horizon + 1)]

    # --------------------------------------------------------------------------------
    # Call from Gazebo
    # --------------------------------------------------------------------------------
    def call_planner(self, v_0: list, b_0: list, visited: list):

        # basic parameters
        self.specs_basic(len(v_0))
        # info from current step
        self.update_for_planner(v_0, b_0, visited)
        # for different simulation configs (pre defined specs according to option)
        self.set_desired_specs()

        # compute plan and danger
        self.hybrid_sim()
        # return things necessary for next iteration

    def output_results(self):
        return self.plan, self.belief_vector, self.visited

    # --------------------------------------------------------------------------------
    # Set specs and current information
    # --------------------------------------------------------------------------------
    def update_for_planner(self, v_0: list, b_0: list, visited: list):
        """Update information that came from Gazebo
        only currently alive searchers"""

        # set team size
        m = len(v_0)
        self.specs_basic(m)
        # set start searchers from current position
        self.specs.set_start_searchers(v_0)
        # set belief calculated from previous time
        self.specs.set_b0(b_0)
        # set vertices visited
        self.visited = visited
        # check to make sure current positions are there
        self.update_visited(v_0)

    def set_desired_specs(self):
        # define extra specs depending on the simulation option desired
        if self.sim_op == 1:
            # danger parameters
            self.specs_danger_common()
        elif self.sim_op == 2:
            self.specs_true_priori()
        elif self.sim_op == 3:
            self.specs_no_constraints()
        elif self.sim_op == 4:
            self.specs_no_danger()
        elif self.sim_op == 5:
            self.specs_prob()
        elif self.sim_op == 6:
            self.specs_true_priori_prob()
        elif self.sim_op == 7:
            self.specs_335()
        elif self.sim_op == 8:
            self.specs_335_prob()
        elif self.sim_op == 9:
            self.specs_new_gt_point()
        elif self.sim_op == 10:
            self.specs_new_gt_prob()
        else:
            exit(print('Please provide a valid sim option.'))

    # --------------------------------------------------------------------------------
    # Specs to run simulations
    # --------------------------------------------------------------------------------

    def specs_basic(self, m=3):

        """Set specs that won't change"""
        # initialize default inputs
        specs = MyInputs2()
        # ------------------------
        # graph number -- SS-2: 8
        specs.set_graph(8)
        # solver parameter: central x distributed
        specs.set_solver_type('distributed')
        # solver timeout (in seconds)
        specs.set_timeout(60)
        # ------------------------
        # time stuff: deadline mission (tau), planning horizon (h), re-plan frequency (theta)
        specs.set_horizon(14)
        specs.set_deadline(100)
        specs.set_theta(1)
        # ------------------------
        # searchers' detection: capture range and false negatives
        specs.set_size_team(m)
        specs.set_capture_range(0)
        specs.set_zeta(None)
        # ------------------------
        # target motion
        specs.set_target_motion('static')
        specs.set_start_target_vertex(None)
        # ------------------------
        # pseudorandom
        # repetitions for each configuration
        specs.set_number_of_runs(1)
        # set random seeds
        self.specs = specs

        return specs

    """sim_op 1"""
    def specs_danger_common(self):
        """Set common danger specs
        """

        # danger files
        base_name = 'danger_map_NCF_freq_'
        # true danger file
        true_file = base_name + '100'
        self.specs.set_danger_file(true_file, 'true')
        # ----------------------------------------
        # estimating danger with 5% images
        # ----------------------------------------
        per = 5
        estimated_file = base_name + str(per).zfill(2)
        # estimated danger file
        self.specs.set_danger_file(estimated_file, 'hat')

        # threshold of searchers

        self.specs.set_threshold(self.kappa, 'kappa')
        self.specs.set_threshold(self.alpha, 'alpha')

        # danger perception
        perception = 'point'
        self.specs.set_danger_perception(perception)

        # Apply prob kill (true/false)
        # hybrid prob (op 3)
        default_prob = 3
        self.specs.set_kill(True, default_prob)
        self.specs.set_mva_conservative(True)
        self.specs.set_use_fov(True)
        self.specs.set_true_estimate(False)

        return self.specs

    """sim_op 2"""
    def specs_true_priori(self):
        # self.specs_danger_common()
        # set perfect a priori knowledge
        self.specs.set_true_know(True)
        # and estimate
        self.specs.set_true_estimate(True)

        return self.specs

    """sim_op 3"""
    def specs_no_constraints(self):

        self.specs.set_kill(True, 3)
        self.specs.set_danger_constraints(False)

        return self.specs

    """sim_op 4"""
    def specs_no_danger(self):
        self.specs_danger_common()
        self.specs.set_kill(False)
        self.specs.set_danger_constraints(False)

        return self.specs

    """sim_op 5"""
    def specs_prob(self):
        self.specs_danger_common()
        # threshold of searchers
        kappa = [3, 4, 5]
        alpha = [0.6, 0.4, 0.4]
        self.specs.set_threshold(kappa, 'kappa')
        self.specs.set_threshold(alpha, 'alpha')

        # danger perception
        perception = 'prob'
        self.specs.set_danger_perception(perception)

    """sim_op 6"""
    def specs_true_priori_prob(self):
        self.specs_prob()

        # set perfect a priori knowledge
        self.specs.set_true_know(True)
        # and estimate
        self.specs.set_true_estimate(True)

    """sim_op 7"""
    def specs_335(self):
        self.specs_danger_common()

        kappa = [3, 3, 5]
        self.specs.set_threshold(kappa, 'kappa')

    """sim_op 8"""
    def specs_335_prob(self):
        self.specs_prob()

        kappa = [3, 3, 5]
        alpha = [0.6, 0.6, 0.4]
        self.specs.set_threshold(kappa, 'kappa')
        self.specs.set_threshold(alpha, 'alpha')

    """sim_op 9"""
    def specs_new_gt_point(self):
        self.specs_basic()

        # danger files
        # true danger file
        true_file = 'gt_danger_NFF'
        self.specs.set_danger_file(true_file, 'true')
        # ----------------------------------------
        # estimating danger with 5% images
        # ----------------------------------------
        per = 5
        estimated_file = 'estimate_danger_fire_des_NFF_freq_05'
        # estimated danger file
        self.specs.set_danger_file(estimated_file, 'hat')

        # threshold of searchers
        kappa = [3, 4, 5]
        alpha = [0.6, 0.4, 0.4]
        self.specs.set_threshold(kappa, 'kappa')
        self.specs.set_threshold(alpha, 'alpha')

        # danger perception
        perception = 'point'
        self.specs.set_danger_perception(perception)

        # Apply prob kill (true/false)
        # hybrid prob (op 3)
        default_prob = 3
        self.specs.set_kill(True, default_prob)
        self.specs.set_mva_conservative(True)
        self.specs.set_use_fov(True)
        self.specs.set_true_estimate(False)

    """sim_op 10"""
    def specs_new_gt_prob(self):
        self.specs_new_gt_point()
        # danger perception
        perception = 'point'
        self.specs.set_danger_perception(perception)

    # --------------------------------------------------------------------------------
    # Simulate and save
    # --------------------------------------------------------------------------------
    def update_visited(self, current_pos: list):

        for v in current_pos:
            if v not in self.visited:
                self.visited.append(v)

        return self.visited

    def hybrid_sim(self):
        specs = self.specs

        # initialize classes
        belief, team, solver_data, target, danger, mission = plnr.init_wrapper(specs)

        # ---------------------------------------
        # Danger stuff
        # ---------------------------------------
        # estimate danger
        danger.estimate(self.visited)

        if self.time_step > 0:

            if danger.kill:
                # compute prob kill, draw your luck and update searchers (if needed)
                team.decide_searchers_luck(danger, self.time_step)
                # update info in danger
                danger.update_teamsize(len(team.alive))
                danger.set_thresholds(team.kappa_original, team.alpha_original)

        # new [check if any searcher is alive]
        if len(team.alive) < 1:
            mission.set_team_killed(True)
            self.set_path_to_output(team.killed)
            return

        # otherwise plan for next time step, considering just the searchers that are alive!
        # -------------------------------------------------------------------------------

        deadline, theta, n = solver_data.unpack_for_sim()
        # deadline, horizon, theta, solver_type, gamma = solver_data.unpack()
        M = target.unpack()

        # initialize time: actual sim time, t = 0, 1, .... T and time relative to the planning, t_idx = 0, 1, ... H
        t, t_plan = 0, 0

        # begin simulation loop
        print('--\nTime step %d \n--' % t)

        # call for planner module
        sim_data = True

        belief, target, team, solver_data, danger, inf = plnr.planner_module(belief, target, team, solver_data,
                                                                             danger, t, sim_data)

        # get planned path as dict:  path [s, t] = v
        path = team.get_path()

        # reset time-steps of planning
        t_plan = 1
        # _________________

        # get dictionary of next positions for searchers, new_pos = {s: v}
        path_next_t = pln.next_from_path(path, t_plan)

        # evolve searcher positions
        # also update visited
        team.searchers_evolve(path_next_t)

        # next time-step
        t, t_plan = t + 1, t_plan + 1

        # retrieve path_next_t (from searchers that are still alive)
        path_next_t = team.retrieve_current_positions()
        # --------------------------------------

        # update belief
        belief.update(team.searchers, path_next_t, M, n)

        # update target
        target = sf.evolve_target(target, belief.new)
        # -------------------------------------------------------------------
        # get things ready to output
        self.set_path_to_output(team.killed, path)
        self.belief_vector = solver_data.retrieve_solver_belief(0, 1)
        self.visited = self.update_visited(team.visited_vertices)

        # # printout for reference
        # team.print_summary()
        print('Vertices visited: %s' % str(self.visited))
        # return belief, target, team, solver_data, danger, mission


if __name__ == '__main__':

    # dummy parameters to test
    n = 46
    v_maybe = [8, 10, 12, 14, 17, 15]
    b_dummy = [0.0 for i in range(n + 1)]
    for v in v_maybe:
        b_dummy[v] = 1./ 6
    # current positions
    # pos_dummy = [2, 3, 4]
    pos_dummy = [1, 1, 1]
    visited_dummy = [1]
    sim_op = 1
    t = 0

    my_sim = MyGazeboSim(pos_dummy, b_dummy, visited_dummy, t, sim_op)
    plan, belief_vector, visited = my_sim.output_results()

    for s in plan.keys():
        print(plan[s])


