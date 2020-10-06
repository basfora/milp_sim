"""Child classes from original milp_mespp classes: added danger-related info"""
from milp_mespp.classes.searcher import MySearcher
from milp_mespp.classes.inputs import MyInputs
from milp_mespp.classes.solver_data import MySolverData
from milp_mespp.core import extract_info as ext
# imports from milp_sim
from milp_sim.risk.classes.danger import MyDanger
import copy


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

        # danger threshold for the searchers
        self.kappa = list()
        self.alpha = list()

        # a priori knowledge of danger
        # eta_priori[v] = [eta_l1,... eta_l5]
        self.danger_priori = []

        # ground truth eta_check[v] \in {1,..5}
        self.danger_true = []

        # type of danger threshold (d - point estimate or p - probabilistic)
        self.perception_list = MyDanger.define_options()
        # default is point estimate
        self.perception = self.perception_list[0]

        self.danger_levels, self.n_levels, self.level_label, self.level_color = MyDanger.define_danger_levels()

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

        self.danger_priori, _ = MyDanger.compute_from_value(self.n)
        self.danger_true, _ = MyDanger.compute_from_value(self.n)

    # UT - ok
    def set_danger_data(self, danger_true, danger_priori):
        """Define true danger and a priori danger
        danger_true: str for file upload, int (same level for all vertices), list of integers,
        or list of lists (prob distributions for each vertex)
        danger_priori: same options, input None for default uniform probability"""

        self.danger_priori = danger_priori
        self.danger_true = danger_true

    def set_danger_perception(self, my_type: str or int):
        """Choose between point, probabilistic or both"""

        if isinstance(my_type, str) and my_type not in self.perception_list:
            print('Error! Danger type not valid, please choose from %s' % str(self.perception_list))
            exit()

        if isinstance(my_type, int) and my_type > 2:
            print('Error! Danger type not valid, please choose from %s' % str(self.perception_list))
            exit()

        self.perception = my_type

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
        else:
            self.alpha = my_list

    def update_n(self):
        self.n = ext.get_set_vertices(self.graph)[1]


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







