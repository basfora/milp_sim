"""Class for Gazebo Sims"""
from milp_sim.risk.classes.child_mespp import MyInputs2
from milp_sim.risk.src import risk_plan as plnr
from milp_mespp.core import plan_fun as pln
from milp_mespp.core import extract_info as ext
import copy


class MyGazeboSim:

    def __init__(self, input_pos: list, b_0: list, visited: list, time_step: int, sim_op=1):

        # original parameters
        self.v0 = [1]
        self.kappa_original = None
        self.alpha_original = None
        self.m_original = 3
        self.S_original = [1, 2, 3]

        # map between original and this call
        self.id_map = []

        # inputs
        self.input_pos = input_pos
        self.b_0 = b_0
        # vertices visited = [v, u, w,..]
        self.visited = visited
        self.time_step = time_step
        self.sim_op = sim_op

        # -----------------------------------------------
        # for this run
        # -----------------------------------------------
        self.m = None
        self.specs = None

        # info used on this call
        self.kappa = None
        self.alpha = None

        # plan[s] = [v0, .... vh]
        self.plan = {}
        # belief_vector = [b_c, b_1,....b_n]
        self.belief_vector = []

        # translate inputs into team information
        self.alive, self.killed, self.current_pos = self.adjust_team()

        # check if we need to abort
        self.abort = self.all_killed(self.alive)

        if not self.abort:
            # specs - set up with new info (adjust team and thresholds)
            self.set_up()

            # call planner
            self.call_planner()

    # --------------------------------------------------------------------------------
    # Adjusting team sizes
    # --------------------------------------------------------------------------------
    def original_thresholds(self):
        self.kappa_original = copy.copy(self.specs.kappa)
        self.alpha_original = copy.copy(self.specs.alpha)

    # UT - ok
    def adjust_team(self):
        """adjust lists based on current positions"""
        alive = []
        # positions of alive searchers
        current_pos = []
        killed = []

        for s in self.S_original:
            s_idx = ext.get_python_idx(s)
            v = self.input_pos[s_idx]

            # if it already died, don't count with it
            if v < 0:
                killed.append(s)
                self.id_map.append((s, -1))
            else:
                alive.append(s)
                current_pos.append(v)
                # new id
                s_new = len(current_pos)
                self.id_map.append((s, s_new))

        return alive, killed, current_pos

    def adjust_threshold(self):
        """"List of threshold of alive searchers"""

        # save original thresholds (from specs)
        self.original_thresholds()

        kappa = []
        alpha = []

        for s_original in self.S_original:
            idx = ext.get_python_idx(s_original)

            if self.is_alive(s_original, self.alive):
                kappa.append(self.kappa_original[idx])
                alpha.append(self.alpha_original[idx])

        self.kappa = kappa
        self.alpha = alpha

        # set thresholds of only of those alive
        self.specs.set_threshold(self.kappa, 'kappa')
        self.specs.set_threshold(self.alpha, 'alpha')

    @staticmethod
    def all_killed(alive_list: list):
        if len(alive_list) < 1:
            return True
        else:
            return False

    @staticmethod
    def is_alive(s_id, alive_list):
        if s_id in alive_list:
            return True
        else:
            return False

    def set_path_to_output(self, just_killed: list, path=None):

        if path is None or self.all_killed(self.alive):
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
        """Return new id for searcher according to id_map"""

        for couple in self.id_map:
            if couple[0] == s_id:
                return couple[1]

    def make_dummy_path(self):
        return [-1 for i in range(self.specs.horizon + 1)]

    # --------------------------------------------------------------------------------
    # Call from Gazebo
    # --------------------------------------------------------------------------------
    def call_planner(self):
        self.hybrid_sim()

    def output_results(self):

        if self.abort:
            print('All robots were lost before this iteration. Simulation was aborted')
            return None, None, None
        else:
            return self.plan, self.belief_vector, self.visited

    # --------------------------------------------------------------------------------
    # Set specs and current information
    # --------------------------------------------------------------------------------
    def set_up(self):
        """Update information that came from Gazebo
        only currently alive searchers"""

        # create basic specs with new team size, belief vector and visited list
        # set team size
        m = len(self.current_pos)
        self.m = m
        # basic specs
        self.specs_basic(m)
        # common danger
        self.specs_danger_common()
        # set start searchers from current position
        self.specs.set_start_searchers(self.current_pos)
        # set belief calculated from previous time
        self.specs.set_b0(self.b_0)
        # make sure current positions are there
        self.update_visited(self.current_pos)

        # for different simulation configs (pre defined specs according to option)
        self.set_desired_specs()

        # need to adjust thresholds for size
        self.adjust_threshold()

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
            self.specs_new_gt_335()
        else:
            exit(print('Please provide a valid sim option. Ending simulation.'))

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
        specs.set_timeout(10)
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
        kappa = [3, 4, 5]
        alpha = [0.95, 0.95, 0.95]
        self.specs.set_threshold(kappa, 'kappa')
        self.specs.set_threshold(alpha, 'alpha')

        # danger perception
        perception = 'point'
        self.specs.set_danger_perception(perception)

        # Apply prob kill (true/false)
        # hybrid prob (op 3)
        default_prob = 3
        self.specs.use_kill(True, default_prob)
        self.specs.set_mva_conservative(True)
        self.specs.set_use_fov(True)
        self.specs.set_true_estimate(False)

        return self.specs

    """sim_op 2"""
    def specs_true_priori(self):

        # set perfect a priori knowledge
        self.specs.set_true_know(True)
        # and estimate
        self.specs.set_true_estimate(True)

        return self.specs

    """sim_op 3"""
    def specs_no_constraints(self):

        self.specs.use_kill(True, 3)
        self.specs.use_danger_constraints(False)

        return self.specs

    """sim_op 4"""
    def specs_no_danger(self):
        self.specs.use_kill(False)
        self.specs.use_danger_constraints(False)

        return self.specs

    """sim_op 5"""
    def specs_prob(self):
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
        # danger files
        # true danger file
        true_file = 'gt_danger_NFF'
        self.specs.set_danger_file(true_file, 'true')
        # ----------------------------------------
        # estimating danger with 5% images
        # ----------------------------------------
        # per = 5
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
        self.specs.use_kill(True, default_prob)
        self.specs.set_mva_conservative(True)
        self.specs.set_use_fov(True)
        self.specs.set_true_estimate(False)

    """sim_op 10"""
    def specs_new_gt_335(self):
        self.specs_new_gt_point()
        # danger perception
        # threshold of searchers
        kappa = [3, 3, 5]
        alpha = [0.6, 0.6, 0.4]
        self.specs.set_threshold(kappa, 'kappa')
        self.specs.set_threshold(alpha, 'alpha')

    # --------------------------------------------------------------------------------
    # Simulate and save
    # --------------------------------------------------------------------------------
    def update_visited(self, current_pos: list):

        for v in current_pos:
            if v not in self.visited:
                self.visited.append(v)

        return self.visited

    def hybrid_sim(self):

        # specs only with new info
        specs = self.specs

        # initialize time: actual sim time, t = 0, 1, .... T and time relative to the planning, t_idx = 0, 1, ... H
        t = self.time_step

        # initialize classes
        belief, team, solver_data, target, danger, mission = plnr.init_wrapper(specs)

        # ---------------------------------------
        # Danger stuff
        # ---------------------------------------
        # estimate danger based on previous visited + current positions
        danger.estimate(self.visited)

        if t > 0 and danger.kill:
            # compute prob kill, draw your luck and update searchers (if needed)
            team.decide_searchers_luck(danger, t)
            # update info in danger
            danger.update_teamsize(len(team.alive))
            danger.set_thresholds(team.kappa, team.alpha)

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

        # begin simulation loop
        print('--\nTime step %d \n--' % t)

        # call for planner module
        sim_data = True

        belief, target, team, solver_data, danger, inf = plnr.planner_module(belief, target, team, solver_data,
                                                                             danger, t, sim_data)

        # break here if the problem was infeasible
        if inf:
            print('Problem was infeasible, stopping simulation.')
            return

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

        # retrieve path_next_t (from searchers that are still alive)
        path_next_t = team.retrieve_current_positions()
        # --------------------------------------

        # update belief
        belief.update(team.searchers, path_next_t, M, n)

        # end of planning
        # -------------------------------------------------------------------
        # get things ready to output
        self.set_path_to_output(team.killed, path)
        self.belief_vector = belief.new

        # printout for reference
        team.print_summary()
        print('Vertices visited: %s \n---' % str(self.visited))




