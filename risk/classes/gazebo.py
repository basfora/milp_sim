"""Class for Gazebo Sims"""
from milp_sim.risk.classes.child_mespp import MyInputs2
from milp_sim.risk.src import plan_risk as plnr, risk_parameters as rp
from milp_mespp.core import plan_fun as pln
from milp_mespp.core import sim_fun as sf, extract_info as ext
import copy


class MyGazeboSim:

    def __init__(self):

        self.id = 0
        self.time_step = 0

        # inputs to the planner
        self.specs = self.specs_basic()

        # outputs for Gazebo
        self.m = self.specs.size_team
        # plan[s] = [v0, .... vh]
        self.plan = {s: [] for s in range(1, self.m+1)}
        # belief_vector = [b_c, b_1,....b_n]
        self.belief_vector = []
        # vertices visited = [v, u, w,..]
        self.visited = []

    def set_for_planner(self, v_0: list, b_0: list, visited: list, t: int):

        # TODO check for inconsistency if m decreases

        m = len(v_0)
        self.specs_basic(m)
        self.specs.set_start_searchers(v_0)
        self.specs.set_b0(b_0)
        self.visited = visited
        self.time_step = t

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
        specs.set_start_seeds(2000, 6000)

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

    def hybrid_sim(self):
        # TODO update/save things accordingly!
        specs = self.specs

        # initialize classes
        belief, team, solver_data, target, danger, mission = plnr.init_wrapper(specs)

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
        team.searchers_evolve(path_next_t)

        # next time-step
        t, t_plan = t + 1, t_plan + 1

        # --------------------------------------
        # new [danger]
        # estimate danger
        danger.estimate(self.visited)

        if danger.kill:
            # compute prob kill, draw your luck and update searchers (if needed)
            team.decide_searchers_luck(danger, t)
            # update info in danger
            danger.update_teamsize(len(team.alive))
            danger.set_thresholds(team.kappa, team.alpha)

        # retrieve path_next_t (from searchers that are still alive)
        path_next_t = team.retrieve_current_positions()
        # --------------------------------------

        # update belief
        belief.update(team.searchers, path_next_t, M, n)

        # update target
        target = sf.evolve_target(target, belief.new)

        # --------------------------------------
        # new [check if searchers are alive]
        if len(team.alive) < 1:
            mission.set_team_killed(True)

        # check for capture based on next position of vertex and searchers
        team.searchers, target = sf.check_for_capture(team.searchers, target)

        mission.save_details(t, copy.copy(team.alive), copy.copy(target.is_captured))
        # printout for reference
        team.print_summary()
        return belief, target, team, solver_data, danger, mission

    def call_planner(self, v_0: list, b_0: list, visited: list, t: int, sim_op=1):

        # basic parameters
        self.specs_basic(len(v_0))
        # info from current step
        self.set_for_planner(v_0, b_0, visited, t)

        if sim_op == 1:
            # danger parameters
            self.specs_danger_common()
        elif sim_op == 2:
            self.specs_true_priori()
        elif sim_op == 3:
            self.specs_100_img()
        else:
            exit(print('Please provide a valid sim option.'))

        # compute plan and danger
        belief, target, team, solver_data, danger, mission = self.hybrid_sim()
        # TODO output updates inside hybrid sim!
        return self.plan, self.belief_vector, self.visited



