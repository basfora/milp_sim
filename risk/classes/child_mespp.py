"""Child classes from original milp_mespp classes: added danger-related info"""
import random

from milp_mespp.classes.searcher import MySearcher
from milp_mespp.classes.inputs import MyInputs
from milp_mespp.classes.solver_data import MySolverData
from milp_mespp.core import extract_info as ext, create_parameters as cp
# imports from milp_sim
from milp_sim.risk.classes.danger import MyDanger
from milp_sim.risk.src import base_fun as bf
import copy
import os
import collections
import datetime

from milp_sim.risk.src import base_fun as bf


class MyInputs2(MyInputs):
    """Child inputs class for milp_sim (risk)
    Parent class: MyInputs from milp_mespp.classes
    needs to store:
    kappa: danger thresholds for each searcher
    alpha: confidence threshold for each searcher"""

    def __init__(self):
        """Need to input:
        eta_priori
        eta_check
        """
        # inherit parent class methods and properties
        super().__init__()

        self.n = 0

        # type of danger threshold (d - point estimate or p - probabilistic)
        self.perception_list = MyDanger.define_options()
        # default is POINT estimate
        self.perception = self.perception_list[0]

        # will change with every interaction
        self.name_folder = ''
        self.path_folder = ''
        d = datetime.datetime.today()
        self.monthday = str(d.month).zfill(2) + str(d.day).zfill(2)

        # configurations
        # danger
        self.danger_constraints = True
        self.danger_kill = True
        self.prob_kill = MyDanger.exp_prob_kill(3)
        # team thresholds
        self.homogeneous_team = False
        # estimate
        self.true_priori = False
        self.true_estimate = False
        self.fov = True
        self.mva_conservative = True
        #
        self.prob_id = 'h'
        # keep
        self.parent_path = ''
        self.set_parent_path()

        # danger threshold for the searchers
        self.kappa = list()
        self.alpha = list()

        # a priori knowledge of danger
        # eta_priori[v] = [eta_l1,... eta_l5]
        self.danger_priori = None

        # ground truth eta_check[v] \in {1,..5}
        self.danger_true = None
        #
        self.danger_hat = None
        self.percentage_img = None

        self.human_gt = False
        self.all_descriptions = False

        self.danger_levels, self.n_levels, self.level_label, self.level_color = MyDanger.define_danger_levels()

    def use_kill(self, status=True, op=3):
        self.danger_kill = status
        self.prob_id = ''

        if status:
            self.set_prob_kill(op)
            if op == 1:
                self.prob_id = 'b'
            elif op == 2:
                self.prob_id = 's'
            elif op == 3:
                self.prob_id = 'h'
        else:
            prob_kill = [0 for i in range(5)]
            self.set_prob_kill(prob_kill)
            self.prob_id = 'n'

    def set_all_descriptions(self, status=False):
        self.all_descriptions = status

    def set_gt(self, status=False):
        self.human_gt = status

    def use_danger_constraints(self, status=True):
        self.danger_constraints = status

    def hh_team(self, status=False):
        self.homogeneous_team = status

    def use_default(self):
        self.update_n()
        self.default_danger_data()
        self.default_thresholds()

    def default_thresholds(self, k=3, alpha=0.95):
        m = self.size_team

        for s in range(m):
            self.kappa.append(k)
            self.alpha.append(alpha)

    def default_danger_data(self):

        self.danger_priori, _ = MyDanger.compute_from_value(self.n, 1, None)
        self.danger_true, _ = MyDanger.compute_from_value(self.n, 1, None)

    # UT - ok
    def set_danger_data(self, data: dict or list, what='true'):
        """Define true danger and a priori danger
        danger_true: str for file upload, int (same level for all vertices), list of integers,
        or list of lists (prob distributions for each vertex)
        danger_priori: same options, input None for default uniform probability"""

        if what == 'true':
            self.danger_true = data
        elif what == 'hat':
            self.danger_hat = data
        elif what == 'priori':
            self.danger_priori = data
        else:
            exit(print('Check your files.'))
        return

    def set_danger_file(self, f_name: str, what='true'):

        if what == 'true':
            self.danger_true = f_name
        elif what == 'hat':
            self.danger_hat = f_name
            per = int(f_name.split('_')[-1])
            self.percentage_img = per
        elif what == 'priori':
            self.danger_priori = f_name
        else:
            exit(print('Check your files.'))
        return

    def set_danger_perception(self, my_type: str or int):
        """Choose between point, probabilistic or both"""

        if isinstance(my_type, str) and my_type not in self.perception_list:
            print('Error! Danger type not valid, please choose from %s' % str(self.perception_list))
            exit()

        if isinstance(my_type, int) and my_type > 2:
            print('Error! Danger type not valid, please choose from %s' % str(self.perception_list))
            exit()

        self.perception = my_type

    def set_true_know(self, status=False):
        self.true_priori = status
        if self.true_estimate or self.true_priori:
            self.percentage_img = 100

    def set_true_estimate(self, status=False):
        self.true_estimate = status
        if status:
            self.danger_hat = None

    def set_mva_conservative(self, status=True):
        self.mva_conservative = status

    def set_use_fov(self, status=True):
        self.fov = status

    # UT - ok
    def set_threshold(self, value: int or list or float, what='kappa'):
        my_list = []
        m = self.size_team

        if isinstance(value, list):
            for el in value:
                my_list.append(el)
        else:
            # same threshold for all searchers, populate for number of searchers
            my_list = [value for s in m]

        if what == 'kappa':
            self.kappa = my_list
            self.homogeneous_team = self.is_homogeneous(my_list)
        else:
            self.alpha = my_list

    def update_n(self):
        self.n = ext.get_set_vertices(self.graph)[1]

    def set_parent_path(self):
        # path to pickle file to be created
        sub_folder = 'exp_data'
        parent_path = bf.get_folder_path('milp_sim', sub_folder, 'risk')
        self.parent_path = parent_path

    def set_name_folder(self, i=1):

        # gate date + turn (i)
        my_date = self.monthday + '_' + str(i).zfill(3)
        my_graph = (self.graph['name'].split('.'))[0]
        my_graph = my_graph.split('_')[0] + my_graph.split('_')[1]

        my_code = []

        if self.danger_constraints is False:
            my_code.append('NC')
        else:
            my_code.append('DC')

        if self.danger_kill is False:
            my_code.append('NK')
        else:
            my_code.append('DK')

        unique = []
        for k in self.kappa:
            if k not in unique:
                unique.append(k)

        if self.homogeneous_team is True:
            my_code.append('HH')
        else:
            my_code.append('HT')

        if self.human_gt is True:
            my_code.append('_HGT')
        else:
            my_code.append('_MGT')

        if self.all_descriptions:
            my_code.append('desFC')
        else:
            my_code.append('desFF')

        if self.danger_constraints:
            my_code.append('_' + str(self.percentage_img).zfill(2))
        else:
            my_code.append('_NA')

        if self.danger_kill is True:
            my_code.append(self.prob_id)

        my_code.append(self.perception)

        if self.fov is False:
            my_code.append('_NFOV')

        if self.true_priori:
            my_code.append('_PK')

        my_name = self.assemble_name(my_code)

        name_folder = my_name + '_' + my_graph + '_' + my_date
        # name of this folder, eg smoke_G9V_grid_date#_run#
        self.name_folder = name_folder

    @staticmethod
    def assemble_name(list_str: list):
        my_name = ''

        for el in list_str:
            my_name += el
        return my_name

    def create_danger_folder(self, turn=1):

        # set the name of the folder
        self.set_name_folder(turn)

        # get full path
        folder_path = self.parent_path + '/' + self.name_folder
        ext.path_exists(folder_path)
        self.path_folder = folder_path

    def set_prob_kill(self, prob_list: list or int):

        if isinstance(prob_list, int):
            prob_list = MyDanger.exp_prob_kill(prob_list)

        self.prob_kill = prob_list

    @staticmethod
    def is_homogeneous(kappa_list: list):

        unique = []
        for k in kappa_list:
            if k not in unique:
                unique.append(k)

        if len(unique) == 1:
            return True
        else:
            return False

    @staticmethod
    def pick_random_belief(n=46, my_seed=None, v0=1):

        areas = bf.compartments_ss2()

        double_v = [1, 3, 4, 9]

        v_possible = []
        for each in areas.keys():

            list_v = areas.get(each)
            # quantity of nodes per area
            q = 1

            if v0 in list_v:
                list_v.pop(list_v.index(v0))

            if each in double_v:
                q = 2

            if my_seed is None:
                v_picked = random.choices(list_v, k=q)
            else:
                v_picked = cp.pick_pseudo_random(list_v, my_seed, q)

            for v in v_picked:
                v_possible.append(v)

        prob = round(1 / len(v_possible), 4)
        nu = n + 1
        b0 = [0.0 for i in range(nu)]
        for v in v_possible:
            b0[v] = prob

        print('Belief vertices: ' + str(v_possible))

        return b0

    def prep_next_turn(self, turn):
        # create folder to store data
        self.create_danger_folder(turn + 1)

        # set seed according to run #
        self.set_seeds(turn)
        # set new belief
        n = 46
        b0 = MyInputs2.pick_random_belief(n, self.target_seed)
        self.set_b0(b0)

        return b0


class MySolverData2(MySolverData):
    # TODO fix this to save what I need

    def __init__(self, horizon, deadline, theta, my_graph, solver_type='central', timeout=30*60):
        # inherit parent class methods and properties
        super().__init__(horizon, deadline, theta, my_graph, solver_type, timeout)

        self.danger_true = list()
        self.eta_hat = dict()
        self.eta = dict()

    def store_risk_vars(self, eta, eta_hat, t):

        self.eta[t] = eta
        self.eta_hat[t] = eta_hat


class MyMission:
    """Class to save key details of the mission (to be expanded later)"""

    def __init__(self):

        # --------------
        # sim configuration (call when initializing class)
        # --------------

        # team info
        self.m = 0
        self.kappa = []
        self.alpha = []
        self.deadline = 100
        self.horizon = 0

        # Danger parameters
        self.homogeneous_team = False
        self.danger_kill = True
        self.danger_constraints = True
        self.prob_kill = []
        self.perception = 'point'
        # not that important anymore (fixing 25%)
        self.fraction_img = 5

        # -------------------
        # other metrics
        # -------------------
        # casualties
        self.casualties = 0
        # save the id of the searchers that died
        self.casual_ids = []
        # save the threshold of those searchers
        self.casual_kappa = []
        # keep list of true (live) and false (killed)
        self.is_alive = []

        # last time step
        self.last_t = 0

        # ----------------
        # End of mission status
        # ----------------
        # False for failure, True for success
        self.success = False

        # target capture - success
        self.target_capture = False

        # deadline reached - failure
        self.deadline_reached = False

        # team was killed - failure
        self.team_killed = False
        self.alive_list = []

    def set_success(self, success=False):
        self.success = success

    def set_team_killed(self, killed=False):
        self.team_killed = killed
        self.set_success(False)

    def add_casualty(self, s_id: int):
        # update the count
        self.casualties += 1
        # id of searcher
        self.casual_ids.append(s_id)
        # threshold
        k = self.kappa[s_id - 1]
        self.casual_kappa.append(k)

    def set_estimate(self, percentage: float):
        self.fraction_img = percentage

    def save_details(self, t: int, alive: list, target_captured: bool):

        # casualties
        self.set_alive(alive)
        # target captured
        self.set_captured(target_captured)
        self.set_success(target_captured)

        # end of mission
        self.set_end(t)
        if self.success is False and self.last_t == self.deadline:
            self.set_reached_deadline(True)

        self.print_details()

    def print_details(self):
        if self.team_killed:
            print('Mission failed, all searchers were killed.')
        elif self.deadline_reached:
            print('Mission failed, target was not found within deadline.')
        else:
            print('Mission succeed, target was found at time %d.' % self.last_t)

    def set_end(self, t: int):
        self.last_t = t

        if t == self.deadline and self.success is False:
            self.set_reached_deadline(True)

    def set_captured(self, capture=False):
        self.target_capture = capture

    def set_reached_deadline(self, status=False, deadline=100):
        self.deadline = deadline
        self.deadline_reached = status
        self.set_success(False)

    def set_perception(self, op: str, img=25):
        self.perception = op
        self.fraction_img = bf.smart_division(img, 100, 2)

    def set_alive(self, alive: list):
        self.alive_list = alive

        if len(self.alive_list) < 1:
            self.set_team_killed(True)
            self.is_alive = [False for s in range(self.m)]
        else:
            is_alive = []
            for s in range(1, self.m + 1):
                if s in self.alive_list:
                    is_alive.append(True)
                else:
                    is_alive.append(False)
                    self.add_casualty(s)
            self.is_alive = is_alive

    def set_deadline(self, deadline: int, horizon=14):
        self.deadline = deadline
        self.horizon = horizon

    def set_team_size(self, m: int):
        self.m = m

        for s in range(m):
            self.is_alive.append(True)

    def set_team_thresholds(self, kappa: list):
        self.kappa = kappa

        self.homogeneous_team = MyInputs2.is_homogeneous(kappa)

    def define_danger(self, constraints=True, kill=True, prob_kill=None):
        self.danger_kill = kill
        if not kill:
            self.prob_kill = []
        else:
            self.prob_kill = prob_kill
        self.danger_constraints = constraints


class MySearcher2(MySearcher):

    """Child searcher class for milp_sim (risk)
    Parent class: MySearcher from milp_mespp.classes"""

    def __init__(self, searcher_id: int, v0: int, g, capture_range=0, zeta=None, my_seed=None):
        # note: init overrides parent class init

        # inherit parent class methods and properties
        super().__init__(searcher_id, v0, g, capture_range, zeta, my_seed)

        self.danger_levels = MyDanger.define_danger_levels()[0]

        # initial thresholds
        # danger threshold
        self.kappa = 3
        self.alpha = 0.95

        # type of danger threshold (d - deterministic or p - probabilistic)
        self.danger_perception = None

        # store danger and beliefs values for vertices visited
        self.path_kappa = dict()

        # life status
        self.alive = True
        self.mva = False

        # original id
        self.id_0 = searcher_id

    def set_kappa(self, new_kappa):
        """Update kappa for searcher"""
        self.kappa = new_kappa

    def set_alpha(self, new_alpha):
        self.alpha = new_alpha

    def set_alive(self, status=True):
        """Set life status: alive (true) or dead (false)"""
        self.alive = status

    def set_new_id(self, s_id):
        self.id = s_id

    def set_perception(self, op: str):
        self.danger_perception = op

    def set_mva(self, status=True):
        self.mva = status

