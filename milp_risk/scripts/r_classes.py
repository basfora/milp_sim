from milp_mespp.classes.searcher import MySearcher
from milp_mespp.classes.inputs import MyInputs
from milp_mespp.core import extract_info as ext


class MySearcher2(MySearcher):

    """Child searcher class for milp_sim (risk)
    Parent class: MySearcher from milp_mespp.classes"""

    def __init__(self, searcher_id: int, v0: int, g, kappa: int or list, rho: float, danger_perception='point',
                 prob_default=0.5, capture_range=0, zeta=None, my_seed=None):
        # note: init overrides parent class init

        # inherit parent class methods and properties
        super().__init__(searcher_id, v0, g, capture_range, zeta, my_seed)

        self.danger_levels = [1, 2, 3, 4, 5]

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

