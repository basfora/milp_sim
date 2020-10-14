"""Class for Gazebo Sims"""
from milp_sim.risk.classes.child_mespp import MyInputs2
from milp_sim.risk.src import plan_risk as plnr, risk_parameters as rp
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
        self.S_new = []

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
        self.plan = {s: [] for s in self.S_original}
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

        s_new = 1
        s = 1
        for v in v_current:
            if v < 0:
                killed.append(s)
            else:
                alive.append(s)
                v0.append(v)
                self.S_new.append(s_new)
                s_new += 1
            s += 1

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

        if path is not None:
            path_as_list = ext.path_as_list(path)
        else:
            path_as_list = path

        s_idnew = 0
        for s_id in self.S_original:
            if s_id in self.killed:
                self.plan[s_id] = [-1 for i in range(self.specs.horizon + 1)]
            else:
                s_idnew += 1
                if s_idnew in just_killed:
                    self.plan[s_id] = [-1 for i in range(self.specs.horizon + 1)]
                else:
                    self.plan[s_id] = path_as_list[s_idnew]
        return

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
            self.specs_100_img()
        else:
            exit(print('Please provide a valid sim option.'))

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
        # TODO check this
        # specs.set_start_seeds(2000, 6000)

        self.specs = specs

        return specs

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

    def specs_true_priori(self):
        # self.specs_danger_common()
        # set perfect a priori knowledge
        self.specs.set_true_know(True)
        # and estimate
        self.specs.set_true_estimate(True)

        return self.specs

    def specs_no_constraints(self):

        self.specs.set_kill(True, 3)
        self.specs.set_danger_constraints(False)

        return self.specs

    def specs_100_img(self):
        # danger files
        base_name = 'danger_map_NCF_freq_'
        # true danger file
        true_file = base_name + '100'
        self.specs.set_danger_file(true_file, 'true')
        # ----------------------------------------
        # estimating only with 5% images
        # ----------------------------------------
        per = 100
        estimated_file = base_name + str(per).zfill(2)
        # estimated danger file
        self.specs.set_danger_file(estimated_file, 'hat')

        return self.specs

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
        # get things ready to be output
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
        b_dummy[v] = 1. / 6
    # current positions
    pos_dummy = [2, -1, 4]
    # pos_dummy = [1, 1, 1]
    visited_dummy = [1, 3]
    sim_op = 1
    t = 3

    my_sim = MyGazeboSim(pos_dummy, b_dummy, visited_dummy, t, sim_op)
    plan, belief_vector, visited = my_sim.output_results()


