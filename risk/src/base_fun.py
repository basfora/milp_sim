import os
import pickle
from milp_mespp.core import extract_info as ext


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
        data = pickle.load(open(file_path, "rb"))

    except:
        print('Make sure your parameters are right!')
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


def is_list(data: dict or list):
    data_list = None
    if isinstance(data, dict):
        data_list = dict_to_list(data)
    elif isinstance(data, list):
        data_list = data

    else:
        exit(print('Data is neither dict or list.'))

    return data_list






