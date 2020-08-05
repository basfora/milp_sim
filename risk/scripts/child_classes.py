"""Child classes from original milp_mespp classes: added danger-related info"""
from milp_mespp.classes.searcher import MySearcher
from milp_mespp.classes.inputs import MyInputs
from milp_mespp.classes.solver_data import MySolverData
from milp_mespp.core import extract_info as ext
from milp_sim.risk.scripts.danger import MyDanger


class MySearcher2(MySearcher):

    """Child searcher class for milp_sim (risk)
    Parent class: MySearcher from milp_mespp.classes"""

    def __init__(self, searcher_id: int, v0: int, g, kappa: int or list, rho: float, danger_perception='point',
                 prob_default=0.5, capture_range=0, zeta=None, my_seed=None):
        # note: init overrides parent class init

        # inherit parent class methods and properties
        super().__init__(searcher_id, v0, g, capture_range, zeta, my_seed)

        self.danger_levels = MyDanger.define_danger_levels()[0]

        # initial thresholds
        # danger threshold
        self.kappa_0 = kappa
        # target belief threshold
        self.rho_0 = rho

        # for storage
        self.stored_kappa = dict()
        self.stored_rho = dict()
        self.stored_rho[0] = self.rho_0

        # type of danger threshold (d - deterministic or p - probabilistic)
        self.danger_perception = danger_perception
        # start danger thresholds
        self.init_kappa(prob_default)

        # current (updated throughout sim)
        self.kappa = kappa
        self.rho = rho

        # store danger and beliefs values for vertices visited
        # path_rho[t] = value
        self.path_kappa = dict()
        self.path_rho = dict()

    def adjust_rho(self, b_c: float):
        """Adjust target belief threshold based on current belief of capture
        b_c = 0, rho = rho_0
        b_c = x, rho  = rho_0 x b_c"""

        new_rho = self.rho_0 * (1 - b_c)
        self.set_new_rho(new_rho)
        self.store_rho(new_rho)

        return new_rho

    def init_kappa(self, prob_default=0.5):
        """Initialize the danger thresholds according to type"""

        my_kappa = self.kappa_0

        if isinstance(self.kappa_0, int):
            my_kappa = []
            for level in self.danger_levels:
                if level <= self.kappa_0:
                    my_kappa.append(1.0)
                else:
                    if self.danger_perception == 'pdf':
                        my_kappa.append(prob_default)
                    else:
                        my_kappa.append(0.0)

        self.stored_kappa[0] = my_kappa

    def set_new_kappa(self, new_kappa):
        """Update kappa - danger belief value"""
        self.kappa = new_kappa

    def set_new_rho(self, new_rho):
        """Update rho - target belief value"""
        self.rho = new_rho

    def store_kappa(self, new_kappa):
        # get current time and vertex
        current_time, current_kappa = ext.get_last_info(self.stored_kappa)
        self.stored_kappa[current_time] = new_kappa

    def store_rho(self, new_rho):
        # get current time and vertex
        current_time, current_rho = ext.get_last_info(self.stored_rho)
        self.stored_rho[current_time] = new_rho

    def store_thresholds(self, new_kappa, new_rho):
        """Store thresholds for this time """
        self.store_kappa(new_kappa)
        self.store_rho(new_rho)


class MyInputs2(MyInputs):
    """Child inputs class for milp_sim (risk)
    Parent class: MyInputs from milp_mespp.classes"""

    def __init__(self):
        # inherit parent class methods and properties
        super().__init__()

        self.rho = None
        self.kappa = None

        # type of danger threshold (d - deterministic or p - probabilistic)
        self.perception_list = MyDanger.define_danger_perceptions()
        self.danger_perception = self.perception_list[0]
        self.prob_default = 0.5

        self.danger_levels, self.n_levels, self.level_label, self.level_color = MyDanger.define_danger_levels()[0]

    def set_danger_perception(self, my_type: str):

        if my_type not in self.perception_list:
            print('Error! Danger type not valid, please choose from %s' % str(self.perception_list))
            exit()

        self.danger_perception = my_type

    def set_danger_threshold(self, value: int or list):
        """Set danger thresholds for each searcher
        :param value :
        if int --> assign the same for all searchers
        if list --> assign each element to a searchers"""

        my_list = []
        m = self.size_team

        # same threshold for all searchers
        if isinstance(value, int):
            # populate for number of searchers
            my_list = [value for s in m]

        elif isinstance(value, list):

            # point estimate (each element in the list corresponds to one searcher)
            if self.danger_perception == self.perception_list[0]:
                # check if all searchers received a threshold
                self.check_size(value, m)
                my_list = value

            # pdf
            elif self.danger_perception == self.perception_list[1]:

                # list of list (first element)
                if isinstance(value[0], list):
                    # repeat that list for all searchers
                    my_list = [value[0] for s in m]
                else:
                    # pdf, list of different thresholds for each searcher
                    for s in range(m):
                        my_list.append(value[s])

        else:
            print('Error! Only accepts int or list')
            exit()

        self.kappa = my_list

    def set_target_threshold(self, my_rho: float or list):

        my_list = []
        if isinstance(my_rho, list):
            self.check_size(my_rho, self.size_team)
            my_list = my_rho
        else:
            for s in range(self.size_team):
                my_list.append(my_rho[s])

        self.rho = my_list

    @staticmethod
    def check_size(my_list, team_size):
        len_list = len(my_list)
        # check if it's point estimate, but wrong number of thresholds was given
        if len_list != team_size:
            print('Error! There are %d searchers, but %d thresholds. Please verify and run again.'
                  % (team_size, len_list))
            exit()

        return


class MySolverData2(MySolverData):

    def __init__(self, horizon, deadline, theta, my_graph, solver_type='central', timeout=30*60):
        # inherit parent class methods and properties
        super().__init__(horizon, deadline, theta, my_graph, solver_type, timeout)

        self.eta = dict()
        self.rho = dict()
        self.u = dict()
        self.w = dict()
        self.A = dict()

    def store_risk_vars(self, eta, rho, u, w, t, A=1):

        self.eta[t] = eta
        self.rho[t] = rho
        self.u[t] = u
        self.w[t] = w
        self.A[t] = A







