import os
import pickle
from milp_mespp.core import extract_info as ext
import numpy as np


def get_scripts_path():
    """Return path of milp_mespp/core directory"""
    # this file directory
    dir_path = os.path.dirname(os.path.abspath(__file__))

    return dir_path


def get_project_path(proj_name='milp_sim'):
    # TODO fix this for other project
    project_level = False
    project_path = None
    this_dir = get_scripts_path()

    while project_level is False:
        end_of_path = this_dir.split('/')[-1]

        if proj_name not in end_of_path:
            this_dir = os.path.dirname(this_dir)
        else:
            project_path = this_dir
            project_level = True

    return project_path


def get_folder_path(proj_name, folder_name, inter_folder=None):
    """Return folder path
    proj_name > folder_name"""

    project_path = get_project_path(proj_name)

    if inter_folder is not None:
        sub_folder = inter_folder + '/' + folder_name
    else:
        sub_folder = folder_name

    folder_path = project_path + '/' + sub_folder

    return folder_path


# default parameters for specs


def check_size(my_list, team_size):
    len_list = len(my_list)
    # check if it's point estimate, but wrong number of thresholds was given
    if len_list != team_size:
        print('Error! There are %d searchers, but %d thresholds. Please verify and run again.'
              % (team_size, len_list))
        exit()

    return


# -------------------------------
# environment related functions (ss_2)
# --------------------------------
def compartments_ss2(n=46):
    """Create lazy FOV - hard code """

    rooms = dict()

    gym = list(range(1, 7))
    h1 = [7, 9, 11]
    h2 = [13, 16, 19, 22, 25, 28, 31, 34, 37, 38, 39, 40]
    abc = [8, 10, 12]
    d = [14, 15, 17, 18]
    e = [20, 21, 23, 24]
    f = [26, 27, 29, 30]
    g = [32, 33, 35, 36]
    cafe = list(range(41, 47))

    rooms[1] = gym
    rooms[2] = h1
    rooms[3] = h2
    rooms[4] = abc
    rooms[5] = d
    rooms[6] = e
    rooms[7] = f
    rooms[8] = g
    rooms[9] = cafe

    return rooms


def fov_ss2():
    # todo TUPLE with vertex (#, distance)

    fov = dict()
    rooms = compartments_ss2()

    areas = list(range(1, len(rooms.keys()) + 1))
    # retrieve graph
    g = ext.get_graph_08()

    for a in areas:
        for v in rooms[a]:
            visible = [u for u in rooms[a] if u != v]
            visible_distance = []

            for u in visible:
                d = ext.get_node_distance(g, v, u)
                visible_distance.append((u, d))

            fov[v] = visible_distance
            if v in [8, 10, 12]:
                fov[v] = []

    return fov


# --------------------------
# load data
# -------------------------
def assemble_file_path(folder_path: str, file_name: str, extension: str):
    file_path = folder_path + '/' + file_name + '.' + extension

    return file_path


def load_pickle_file(file_path: str):
    """Load pickle file, given complete file path"""

    try:
        with open(file_path, "rb") as f:
            data = pickle.load(f)

    except:
        print('Make sure your parameters are right! %s does not exist' % file_path)
        data = None

    return data


def dict_to_list(data_dict: dict):
    """Convert data from a dictionary into a list (of lists if necessary)"""

    keys = [key for key in data_dict.keys()]

    data_list = []

    for k in keys:
        # get information
        data = data_dict.get(k)
        # append to list
        data_list.append(data)

    return data_list


def dict_array_to_list(data_dict: dict):
    """Convert data from a dictionary into a list (of lists if necessary)"""

    keys = [key for key in data_dict.keys()]

    data_list = []

    for k in keys:
        # get information
        data = list(data_dict.get(k))
        # append to list
        data_list.append(data)

    return data_list


def is_list(data: dict or list):
    data_list = None
    if isinstance(data, dict):
        k = [key for key in data.keys()]
        if isinstance(data[k[0]], list):
            data_list = dict_to_list(data)
        else:
            data_list = dict_array_to_list(data)
    elif isinstance(data, list):
        data_list = data

    else:
        exit(print('Data is neither dict or list.'))

    return data_list


def list_to_dict(data: list):

    n = len(data)
    V = ext.get_set_vertices(n)[0]

    data_dict = {}
    for v in V:
        vidx = ext.get_python_idx(v)
        data_dict[v] = data[vidx]

    return data_dict


def make_pickle_file(data, f_path: str):

    my_pickle = open(f_path, "wb")
    pickle.dump(data, my_pickle)
    my_pickle.close()

    folder_name = f_path.split('/')[-2]
    file_name = f_path.split('/')[-1]
    print_name = folder_name + ' - ' + file_name

    print("Data saved in: ", print_name)
    return


def make_dir(folder_path: str):

    ext.path_exists(folder_path)
    return


def save_sim_data(belief, target, team, solver_data, danger, specs, mission, file_name='saved_data'):

    # path to pickle file to be created
    # sub_folder = 'exp_data'
    # parent_path = get_folder_path('milp_sim', sub_folder, 'risk')
    # folder_path = parent_path + '/' + name_folder
    #
    # make_dir(folder_path)

    folder_path = specs.path_folder
    file_path = assemble_file_path(folder_path, file_name, 'pkl')

    data = dict()
    data['belief'] = belief
    data['target'] = target
    data['team'] = team
    data['solver_data'] = solver_data
    data['danger'] = danger
    data['specs'] = specs
    data['mission'] = mission

    make_pickle_file(data, file_path)

    if os.path.exists(file_path):
        return True


def save_log_file(path_list, belief_nonzero, plan_eval, danger_ok, danger_error, my_config, t, folder_path):
    """Save log in txt file if needed to debug"""

    file_name = 'sim_log'
    file_path = assemble_file_path(folder_path, file_name, 'txt')

    f = open(file_path, 'a')

    lines = dict()
    lines[0] = '---------'
    lines[1] = str(t) + '\n'
    lines[2] = str(my_config) + '\n'
    lines[3] = str(belief_nonzero) + '\n'
    lines[4] = str(path_list) + '\n'
    lines[5] = str(plan_eval) + '\n'
    lines[6] = str(danger_ok) + '\n'
    lines[7] = str(danger_error) + '\n'

    for k in lines:
        line = lines[k]
        f.write(line)

    f.close()

    return


# ------------------------
# handy functions
# -----------------------
def smart_min(in_data):
    """"""
    if isinstance(in_data, int):
        print('Integer input, returning it.')
        return in_data
    else:
        if len(in_data) < 1:
            print('Empty list, returning None.')
            return None
        else:
            return min(in_data)


def smart_max(in_data):
    if isinstance(in_data, int):
        print('Integer input, returning it.')
        return in_data
    else:
        if len(in_data) < 1:
            print('Empty list, returning None.')
            return None
        else:
            return max(in_data)


def smart_division(in_data, dem=100, dec=2):

    result = None

    if isinstance(in_data, int):
        result = divide(in_data, dem, dec)
    elif isinstance(in_data, list):
        result = []
        for el in in_data:
            result.append(divide(el, dem, dec))
    elif isinstance(in_data, dict):
        result = {}
        for k in in_data.keys():
            result[k] = divide(in_data[k], dem, dec)
    elif in_data is None:
        result = None
    else:
        print('Division error, %s / %s' % (str(in_data), str(dem)))

    return result


def smart_in(v, my_list):

    is_in = False

    if isinstance(my_list, list):

        if isinstance(my_list[0], list):
            for inner_list in my_list:
                if v in inner_list:
                    is_in = True
                    break
        else:
            if v in my_list:
                is_in = True
            else:
                is_in = False

    return is_in


def divide(num, dem, dec):
    return round(num / dem, dec)


def next_position(path: dict or list):
    next_pos = []

    if isinstance(path, dict):
        for s in path.keys():
            next_pos.append(path[s][0])
    elif isinstance(path, list):
        for s in path:
            next_pos.append(s[0])

    return next_pos


def smart_list_add(list1: list, list2: list):
    for el in list2:
        if el not in list1:
            list1.append(el)

    return list1





