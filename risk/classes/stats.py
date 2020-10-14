from matplotlib import pyplot as plt
# import all classes
from milp_sim.risk.classes.danger import MyDanger
from milp_sim.risk.classes.child_mespp import MyInputs2, MyMission, MySolverData2
from milp_sim.risk.classes.team import MyTeam2
# other imports
from milp_sim.risk.src import base_fun as bf
from milp_sim.scenes import make_graph as mg, files_fun as ff
from milp_sim.risk.src import r_plotfun as rpf
from collections import Counter
import copy
import pickle


plt.rcParams.update({'font.size': 16})
plt.rcParams.update({'legend.fontsize': 12})
# ieee compliant plots
plt.rcParams.update({'text.usetex': True})
plt.rcParams.update({'font.family': 'serif'})


class MyStats:
    """Class to make computing stats easier for each configuration (e.g. NC DK worst case scenario, big prob)
    """

    def __init__(self, folder_name: str, instance_base: str, n_runs=1000): # perception: str, n_runs=1000, probname='h', etra=''):

        # to id
        self.parent_name = folder_name
        self.instance_base = instance_base

        # config name, e.g. NCDK
        self.config = folder_name.split('-')[0][4:]
        self.img_per = folder_name.split('-')[-1]

        # self.img_per = self.get_img_per(img_per)
        self.print_name(folder_name, '--\nStats for')
        self.print_name(self.img_per, 'Percentage images: ', False)

        # # path to parent folder (e.g. 1010DCDK-b)
        # self.date_name = folder_name[0:4]
        # self.p_type = folder_name.split('-')[-2]
        # self.extra = folder_name.split('-')[-1]
        self.parent_path = self.get_parent_path(folder_name)

        # to save
        self.save_path = self.get_save_path(folder_name + '-' + self.img_per)

        # common
        self.graph_name = 'G46Vss2'
        # self.perception = perception
        self.middle_name = instance_base
        # self.assemble_middle()
        self.f_name = 'saved_data.pkl'

        self.n_runs = 0
        self.list_runs = []
        self.set_n_turns(n_runs)

        # data related stuff
        self.instances_paths = []
        self.instances_names = []
        self.instances = []

        # -----------------------------------------
        # Parameters - common to all instances in this config
        # danger parameters
        self.prob_kill = []
        self.danger_kill = True
        self.danger_constraints = True
        # team parameters
        self.m = 0
        self.team_hh = False
        self.kappa = []
        # time parameters
        self.horizon = 0
        self.deadline = 0
        self.timeout = 0
        self.env = 'NFC'
        # target parameter
        self.target_motion = 'static'

        # --------------------
        # Actual Stats
        # --------------------
        """MISSION SUCCESS"""
        # collected
        # list of successful missions
        self.success_list = list()
        # counter
        self.success_counter = 0
        # computed metrics
        self.success_rate = 0.0

        """MISSION FAILURE"""
        # collected
        # list of successful missions
        self.fail_list = list()
        self.fail_counter = 0
        # computed metrics
        self.fail_rate = 0.0
        # failure due to abort
        self.abort_list = []
        self.abort_counter = 0
        self.abort_rate = 0.0
        # failure due to to deadline reached
        self.cutoff_list = []
        self.cutoff_counter = 0
        self.cutoff_rate = 0.0

        """CASUALTIES"""
        # lists
        self.casualty_mvp = []
        # number of casualties per mission, list[id] = 0/1/2/3
        self.casualty_list = []

        # in how many missions the MVP died, +1 for each mission it happened
        self.casualty_counter_mvp = 0
        # missions with any casualties
        self.casualty_counter_any = 0
        # just sum 0/+1/+2/+3
        self.casualty_counter = 0

        # computed: casualty_rate_MVP = counter_mvp/missions
        self.casualty_rate_mvp = 0.0
        self.casualty_rate_not_mvp = 0.0
        # computed : casualty rate = casualty_total/(missions * searchers)
        self.casualty_rate = 0.0
        # computed: missions with casualty/missions
        self.casualty_mission = 0.0

        """MISSION TIME"""
        # list of mission end times
        self.mission_time_list = []
        # mission_time_avg = sum(mission_times)/missions
        self.mission_time_avg = 0.0
        # list of abort times (len == self.abort_list)
        self.abort_time_list = []
        # abort_time_avg = sum(abort_times)/abort_counter
        self.abort_time_avg = 0.0
        # capture time list (len == success_list)
        self.capture_time_list = []
        # capture_time_avg = sum(capture_time)/success_counter
        self.capture_time_avg = 0.0

        """STABILITY"""
        # compute states retroactively for 1...1000 instances (to plot)
        self.cum_success_rate = []
        self.cum_fail_rate = []
        self.cum_abort_rate = []
        self.cum_cutoff_rate = []
        self.cum_casualty_mvp_rate = []
        self.cum_casualty_not_mvp_rate = []
        self.cum_casualty_rate = []
        self.cum_casualty_mission_rate = []
        self.cum_mission_time = []
        self.cum_abort_time = []
        self.cum_capture_time = []

        """Collect, compute and save"""
        # collect data
        self.get_stats()

        # print stats
        self.print_final_stats()

        # save data
        self.save_all_data()

    """Called on init"""
    @staticmethod
    def get_parent_path(parent_name: str):
        data_saved_path = MyDanger.get_folder_path('data_saved')
        parent_path = data_saved_path + '/' + parent_name
        return parent_path

    @staticmethod
    def get_save_path(f_name: str):
        data_compiled_path = MyDanger.get_folder_path('data_compiled')
        save_path = data_compiled_path + '/' + f_name
        return save_path

    def assemble_middle(self):
        prob = self.p_type
        add = '_' + self.extra

        graphname = prob + self.perception + add +'_G46Vss2_'
        my_name = '_' + str(self.img_per).zfill(2) + graphname + self.date_name + '_'

        self.middle_name = my_name

    @staticmethod
    def get_img_per(percentage: int):
        return str(percentage).zfill(2)

    def set_n_turns(self, value: int):
        self.n_runs = value
        self.set_run_list()

    def set_run_list(self):
        self.list_runs = list(range(1, self.n_runs + 1))
        return self.list_runs

    def get_stats(self):
        self.collect_data()

        # final stats
        self.set_final_status()

    """Called during data collection"""
    def assemble_instance_name(self, turn, change_date=None):

        # if change_date is not None:
        #     self.date_name = change_date
        #     self.assemble_middle()

        instance_name = self.instance_base + str(turn).zfill(3)
        self.instances_names.append(instance_name)
        return instance_name

    def assemble_f_path(self, instance_name):
        f_path = self.parent_path + '/' + instance_name + '/' + self.f_name
        self.instances_paths.append(f_path)
        return f_path

    def set_prob_kill(self, prob_list: list):
        self.prob_kill = prob_list

    """Collecting data from instances"""
    def collect_data(self):

        self.print_name(self.config, 'Collecting')
        self.print_dots()
        n_ok = 0

        for n_turn in self.list_runs:

            instance_name = self.assemble_instance_name(n_turn)

            if '1009' in self.parent_name and self.img_per == '05' and n_turn == 104:
                instance_name = self.assemble_instance_name(n_turn, '1010')

            f_path = self.assemble_f_path(instance_name)
            try:
                data = pickle.load(open(f_path, "rb"))
                n_ok += 1
            except:
                print('Trouble collecting instance %d' % n_turn)
                # go to next
                continue

                # save prob kill (equal to all instances)

            # create an instance obj
            instance = MyInstance(n_turn, data)

            if self.deadline == 0:
                self.set_common_parameters(instance)

            # update counters and lists
            self.update_counters(instance)
            # update cum stats
            self.update_cum_stats(instance)
            # save in list for storage
            self.instances.append(copy.deepcopy(instance))

        print('Done, %d/%d instances collected' % (n_ok, self.n_runs))

    def update_counters(self, ins):

        # related to mission status
        if ins.success:
            self.update_success(ins)
        else:
            self.update_fail(ins)

        # update mission time list
        self.mission_time_list.append(ins.end_time)

        # casualty stats
        self.update_casualties(ins)

    def update_success(self, ins):
        self.success_list.append(ins.id)
        self.success_counter += 1
        self.capture_time_list.append(ins.end_time)

    def update_fail(self, ins):
        self.fail_list.append(ins.id)
        self.fail_counter += 1

        if ins.abort:
            self.abort_list.append(ins.id)
            self.abort_counter += 1
            self.abort_time_list.append(ins.end_time)
        else:
            self.cutoff_list.append(ins.id)
            self.cutoff_counter += 1

    def update_casualties(self, ins):

        # casualties on that mission
        was_casualty = copy.deepcopy(ins.casualty)
        c = copy.deepcopy(ins.number_casualties)

        # list
        self.casualty_list.append(c)

        # update total casualties
        self.casualty_counter += c

        # update missions with casualties
        if ins.casualty:
            self.casualty_counter_any += 1

        # update missions with MVP casualty
        if ins.casualty_mvp:
            self.casualty_mvp.append(True)
            self.casualty_counter_mvp += 1
        else:
            self.casualty_mvp.append(False)

    def update_cum_stats(self, ins, q=None):
        if q is None:
            q = ins.id

        # compute stats up to current instance (q)
        # compute cumulative rates
        success_rate = self.get_rate(self.success_counter, q)
        fail_rate = self.get_rate(self.fail_counter, q)
        abort_rate = self.get_rate(self.abort_counter, q)
        cutoff_rate = self.get_rate(self.cutoff_counter, q)

        # update lists
        self.cum_success_rate.append(success_rate)
        self.cum_abort_rate.append(abort_rate)
        self.cum_cutoff_rate.append(cutoff_rate)
        self.cum_fail_rate.append(fail_rate)

        # time related rates
        avg_mission_time = self.get_avg(self.mission_time_list)
        # this might be None
        avg_abort_time = self.get_avg(self.abort_time_list)
        avg_capture_time = self.get_avg(self.capture_time_list)

        # update lists
        self.cum_mission_time.append(avg_mission_time)
        self.cum_abort_time.append(avg_abort_time)
        self.cum_capture_time.append(avg_capture_time)

        self.update_cum_casualties(q)

    def update_cum_casualties(self, q: int):
        """q : number of missions so far"""

        # casualty rate = n_killed/all_n (casualty rate = casualty_total/(missions * searchers))
        casualty_rate = self.get_rate(self.casualty_counter, q * self.m)
        # mvp rate: counter_killed_mvp/missions (casualty_rate_MVP = counter_mvp/missions)
        mvp_rate = self.get_rate(self.casualty_counter_mvp, q)
        # fraction of missions with casualties
        casualty_mission_rate = self.get_rate(self.casualty_counter_any, q)
        not_mvp_rate = self.get_rate(self.casualty_counter_any-self.casualty_counter_mvp, q)

        self.cum_casualty_rate.append(casualty_rate)
        self.cum_casualty_mvp_rate.append(mvp_rate)
        self.cum_casualty_mission_rate.append(casualty_mission_rate)
        self.cum_casualty_not_mvp_rate.append(not_mvp_rate)

    def set_final_status(self):

        self.success_rate = self.get_last(self.cum_success_rate)
        self.abort_rate = self.get_last(self.cum_abort_rate)
        self.cutoff_rate = self.get_last(self.cum_cutoff_rate)
        self.fail_rate = self.get_last(self.cum_fail_rate)

        self.casualty_rate = self.get_last(self.cum_casualty_rate)
        self.casualty_rate_mvp = self.get_last(self.cum_casualty_mvp_rate)
        self.casualty_rate_not_mvp = self.get_last(self.cum_casualty_not_mvp_rate)
        self.casualty_mission = self.get_last(self.cum_casualty_mission_rate)

        self.mission_time_avg = self.get_last(self.cum_mission_time)
        self.abort_time_avg = self.get_last(self.cum_abort_time)
        self.capture_time_avg = self.get_last(self.cum_capture_time)

    """Retrieving stuff"""
    @staticmethod
    def get_rate(in_data: list or int, q: int):
        if isinstance(in_data, list):
            my_rate = round(sum(in_data)/q, 4)
        else:
            my_rate = round(in_data/q, 4)

        return my_rate

    @staticmethod
    def get_avg(in_data: list):
        q = len(in_data)
        if q == 0:
            my_mean = None
        else:
            my_mean = round(sum(in_data)/q, 2)

        return my_mean

    @staticmethod
    def per_from_prob(value: float):
        return round(100 * value, 4)

    @staticmethod
    def get_last(my_list):
        return copy.copy(my_list[-1])

    def set_common_parameters(self, ins):
        # danger parameters
        self.prob_kill = copy.deepcopy(ins.danger.prob_kill)
        self.danger_kill = ins.danger.kill
        self.danger_constraints = ins.danger.constraints

        # sanity check
        if 'DK' in self.config:
            assert self.danger_kill is True
        elif 'NK' in self.config:
            assert self.danger_kill is False
        if 'DC' in self.config:
            assert self.danger_constraints is True
        elif 'NC' in self.config:
            assert self.danger_constraints is False

        # team parameters
        self.m = copy.deepcopy(ins.mission.m)
        self.team_hh = ins.mission.homogeneous_team
        self.kappa = ins.mission.kappa
        # time parameters
        self.horizon = ins.solver_data.horizon[0]
        self.deadline = ins.solver_data.deadline
        self.timeout = ins.solver_data.timeout
        self.env = 'NFC'
        # target parameter
        self.target_motion = 'static'

    """Save data for storage"""
    def save_all_data(self):
        save_path = self.save_path
        extension = '.pkl'
        f_path = save_path + extension
        bf.make_pickle_file(self, f_path)

        self.print_dots()
        self.print_dash()

    """Printing functions"""
    @staticmethod
    def print_dots():
        print('...........')

    @staticmethod
    def print_dash():
        print('-----------\n')

    @staticmethod
    def print_name(f_name: str, action='Collecting', data=True):
        if data:
            s = ' data: '
        else:
            s = ''
        my_str = action + s + f_name
        print(my_str)

    def print_final_stats(self):

        # rates
        self.print_stats('Success', self.success_rate)
        self.print_stats('Fail', self.fail_rate)
        print('..Breakdown')
        self.print_stats('Abort', self.abort_rate)
        self.print_stats('Cutoff', self.cutoff_rate)

        self.print_dots()

        # avg
        self.print_stats('Mission time', self.mission_time_avg, 2)
        self.print_stats('Capture time:', self.capture_time_avg, 2)
        self.print_stats('Abort time', self.abort_time_avg, 2)

        self.print_dots()

        # casualties
        self.print_stats('Casualty', self.casualty_rate)
        self.print_stats('Mission with casualties,', self.per_from_prob(self.casualty_mission), 4)
        print('..Breakdown')
        self.print_stats('Mission with MVP casualty,', self.per_from_prob(self.casualty_rate_mvp), 4)
        self.print_stats('Mission with non-MVP casualty,', self.per_from_prob(self.casualty_rate_not_mvp), 4)

        self.print_dots()

    @staticmethod
    def print_stats(what: str, value: float or int, op=1):
        if op == 1:
            adj = ' rate: '
        elif op == 2:
            adj = ' avg: '
        elif op == 3:
            adj = ' per mission: '
        else:
            adj = ' percentage: '
        my_str = what + adj + str(value)

        print(my_str)


class MyInstance:
    """Save data from this particular instance"""
    def __init__(self, id_number: int, data: dict):

        self.id = id_number

        # just for storage
        self.belief = None
        self.target = None
        self.danger = None
        self.mission = None
        self.team = None
        self.specs = None
        self.solver_data = None

        # mission status and time
        self.success = False
        self.abort = False
        self.cutoff = False
        self.end_time = 0

        # casualties
        self.number_casualties = 0
        self.casualty_mvp = False
        self.casualty = False

        # extract important data
        self.extract_data(data)
        self.set_mission_status()
        self.set_casualties()

    def extract_data(self, data):
        """Get data saved from simulation"""
        self.belief = copy.deepcopy(data['belief'])
        self.target = copy.deepcopy(data['target'])
        self.danger = copy.deepcopy(data['danger'])
        self.mission = copy.deepcopy(data['mission'])
        self.team = copy.deepcopy(data['team'])
        self.specs = copy.deepcopy(data['specs'])
        self.solver_data = copy.deepcopy(data['solver_data'])

    def set_mission_status(self):

        if self.target.is_captured:
            # successful
            self.success = True
            self.end_time = copy.deepcopy(self.target.capture_time)
        else:
            # failed
            self.success = False
            # agents killed
            if len(self.team.alive) < 1:
                self.abort = True
                self.end_time = self.mission.last_t
            elif self.mission.last_t == self.specs.deadline:
                # reached deadline
                self.cutoff = True
                self.end_time = self.mission.deadline
            else:
                exit(print('Check instance %d cause of failure' % self.id))

    def set_casualties(self):

        mvp = 1

        if self.abort is True:
            # number casualties = team size
            self.number_casualties = copy.deepcopy(self.mission.m)
            self.casualty_mvp = True
        else:
            for s in self.mission.casual_ids:
                self.number_casualties += 1
                if s == mvp:
                    self.casualty_mvp = True

        if self.number_casualties > 0:
            self.casualty = True







