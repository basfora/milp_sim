"""Class for Gazebo Sims"""
from milp_sim.risk.classes.child_mespp import MyInputs2
from milp_sim.risk.src import risk_plan as plnr
from milp_mespp.core import extract_info as ext
from milp_sim.risk.exp_src import ral_default as ral
import copy


class MyGazeboSim:

    def __init__(self, input_pos: list, b_0: list, visited: list, time_step: int, sim_op=1, log_path=None):

        # original parameters (tailored to ral)
        self.v0 = [1]
        self.m_original = 3
        self.S_original = [1, 2, 3]
        self.kappa_original, self.alpha_original = None, None

        # map between original and this call
        self.id_map = []

        # -----------------------------------------------
        # inputs from Gazebo
        # -----------------------------------------------

        # position
        self.input_pos = input_pos
        # belief vector
        self.bt_1 = b_0
        # vertices visited = [v, u, w,..]
        self.visited_t_1 = visited
        self.time_step = time_step
        self.sim_op = sim_op
        self.log_path = log_path

        # -----------------------------------------------
        # init empty vars
        # -----------------------------------------------
        self.m = None
        self.specs = None
        self.visited = []

        # to be updated
        self.b_0 = []
        self.dict_pos = {}

        # info used on this call
        self.kappa = None
        self.alpha = None

        # plan[s] = [v0, .... vh]
        self.plan = {}
        # belief_vector = [b_c, b_1,....b_n]
        self.belief_vector = []

        # -----------------------------------------------
        # team info for this run
        # -----------------------------------------------

        # translate inputs into team information
        self.alive, self.killed, self.current_pos = self.adjust_team()

        # check if we need to abort (true or false)
        self.abort = self.all_killed(self.alive)

        if not self.abort:
            # set up specs with new info
            self.set_up()

            # call planner
            self.call_planner()

    # --------------------------------------------------------------------------------
    # Adjusting team sizes
    # --------------------------------------------------------------------------------

    # UT - ok
    def adjust_team(self):
        """adjust lists based on current positions"""
        # list of alive searchers
        alive = []
        # list of killed searchers (either danger or stuck)
        killed = []
        # alive searchers current positions
        current_pos = []

        for s in self.S_original:
            s_idx = ext.get_python_idx(s)
            v = self.input_pos[s_idx]

            # if it already died (v = -1 or v = -2), don't count with it
            if v < 0:
                killed.append(s)
                self.id_map.append((s, -1))
            else:
                alive.append(s)
                current_pos.append(v)
                # new id
                s_new = len(current_pos)
                self.id_map.append((s, s_new))
                # for new ids, assemble dict current_pos = {s: v}
                self.dict_pos[s_new] = v

        return alive, killed, current_pos

    def adjust_threshold(self, specs):
        """"List of threshold of alive searchers"""

        kappa = []
        alpha = []

        # not adjusted for team size
        self.kappa_original = specs.kappa
        self.alpha_original = specs.alpha

        for s_original in self.S_original:
            idx = ext.get_python_idx(s_original)

            if self.is_alive(s_original, self.alive):
                kappa.append(self.kappa_original[idx])
                alpha.append(self.alpha_original[idx])

        self.kappa = kappa
        self.alpha = alpha

        # set thresholds of only of those alive
        specs.set_threshold(self.kappa, 'kappa')
        specs.set_threshold(self.alpha, 'alpha')

        return specs

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
                    # was killed on or before this iteration
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
    # Set specs with current info
    # --------------------------------------------------------------------------------
    def set_up(self):
        """Update information that came from Gazebo
        only currently alive searchers"""

        # create basic specs with new team size, belief vector and visited list
        # set team size
        m = len(self.current_pos)
        self.m = m

        # basic specs
        specs = ral.specs_gazebo_sim(m)

        # set start searchers from current position
        specs.set_start_searchers(self.current_pos)

        # set belief from previous time
        specs.set_b0(self.bt_1)

        # make sure current positions are there
        self.update_visited(self.current_pos)
        # for different simulation configs (pre defined specs according to option)
        self.set_desired_specs(specs)

    def set_desired_specs(self, specs):
        # define extra specs depending on the simulation option desired
        if self.sim_op == 1:
            my_specs = ral.pt_pu_345(specs)
        elif self.sim_op == 2:
            my_specs = ral.pt_pk_345(specs)
        elif self.sim_op == 3:
            my_specs = ral.specs_no_constraints(specs)
        elif self.sim_op == 4:
            my_specs = ral.specs_no_danger(specs)
        elif self.sim_op == 5:
            my_specs = ral.pb_pu_345(specs)
        elif self.sim_op == 6:
            my_specs = ral.pb_pk_345(specs)
        elif self.sim_op == 7:
            my_specs = ral.pt_pu_335(specs)
        elif self.sim_op == 8:
            my_specs = ral.pt_pu_333(specs)
        elif self.sim_op == 9:
            my_specs = ral.pb_pu_335(specs)
        elif self.sim_op == 10:
            my_specs = ral.pb_pu_333(specs)
        else:
            my_specs = None
            exit(print('Please provide a valid sim option. Ending simulation.'))

        # set threshold of searchers
        specs = self.adjust_threshold(my_specs)

        self.specs = specs

    # --------------------------------------------------------------------------------
    # Simulate and save
    # --------------------------------------------------------------------------------
    def update_visited(self, current_pos: list):

        for v in self.visited_t_1:
            self.visited = self.smart_in(v, self.visited)

        for v in current_pos:
            self.visited = self.smart_in(v, self.visited)

        return self.visited

    @staticmethod
    def smart_in(v, list_v):

        if isinstance(v, list):
            for u in v:
                if u not in list_v:
                    list_v.append(u)
        else:
            if v not in list_v:
                list_v.append(v)

        return list_v

    def hybrid_sim(self):

        # specs only with new info
        specs = self.specs

        # initialize time: actual sim time, t = 0, 1, .... T and time relative to the planning, t_idx = 0, 1, ... H
        t = self.time_step

        # initialize classes, with t - 1 info
        belief, team, solver_data, target, danger, mission = plnr.init_wrapper(specs)

        deadline, theta, n = solver_data.unpack_for_sim()
        # current_pos = {s: v}
        M = target.unpack()

        # update belief
        belief.update(team.searchers, self.dict_pos, M, n)

        self.b_0 = belief.new

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

        # need to abort? [check if any searcher is alive]
        if len(team.alive) < 1:
            mission.set_team_killed(True)
            self.set_path_to_output(team.killed)
            return

        # otherwise plan for next time step, considering just the searchers that are alive!
        # -------------------------------------------------------------------------------

        # begin simulation loop
        print('--\nTime step %d \n--' % t)

        # call for planner module
        sim_data = True

        belief, target, team, solver_data, danger, inf = plnr.planner_module(belief, target, team, solver_data,
                                                                             danger, t, sim_data, self.log_path)

        # break here if the problem was infeasible
        if inf:
            print('Problem was infeasible, stopping simulation.')
            return

        # get planned path as dict:  path [s, t] = v
        path = team.get_path()

        # end of planning
        # -------------------------------------------------------------------
        # get things ready to output
        self.set_path_to_output(team.killed, path)
        self.belief_vector = copy.copy(self.b_0)

        # printout for reference
        team.print_summary()
        print('Vertices visited: %s \n---' % str(self.visited))
