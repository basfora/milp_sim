import os


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



