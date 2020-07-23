import os


def get_file_path(name_file, parent_folder='txt_files', my_format='.txt'):
    """Get txt file path inside scenarios directory"""
    # this file directory
    path1 = os.path.dirname(os.path.abspath(__file__))

    path2 = path1 + '/' + parent_folder

    file_path = path2 + '/' + name_file + my_format

    return file_path



def space():
    return ' '


def read_line(line, key=' '):
    """Split line according to key
    Return list of strings"""

    str_list = []
    str_line = line.split(key)
    for el in str_line:
        if "\n" in el:
            el = el.split('\n')[0]
        str_list.append(el)

    return str_list


def read_vertices(file_path: str):
    """Return as dictionary of lists
    vertices[v] = (x, y) """

    f = open(file_path, 'r')

    vertices = dict()
    i = -1
    for line in f.readlines():

        str_list = read_line(line)

        if i == -1:
            vertices['floorplan'] = str_list
        elif i == 0:
            pass
        else:
            v = int(str_list[0])
            vertices[v] = (float(str_list[1]), float(str_list[2]))
        i += 1

    f.close()
    return vertices


def read_file(file_path: str):
    """Return as dictionary of lists
        data[line #] = [str] """

    f = open(file_path, 'r')

    data = {}

    i = 0
    for line in f.readlines():
        i += 1
        data[i] = read_line(line)

    f.close()
    return data


def get_vertices(name_file, parent_folder='txt_files'):
    file_path = get_file_path(name_file, parent_folder)
    V = read_vertices(file_path)
    return V





