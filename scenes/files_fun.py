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
        if not el == '':
            str_list.append(el)

    return str_list


def read_data_txt(file_path: str, my_type='float'):
    """Return as dictionary of lists
    vertices[env] = id
    vertices[v] = (x, y), v = 1,...n """

    f = open(file_path, 'r')

    vertices = dict()
    i = -1
    for line in f.readlines():

        str_list = read_line(line)

        if i == -1:
            vertices['env'] = str_list
        elif i == 0:
            pass
        else:
            v = int(str_list[0])
            if my_type == 'float':
                vertices[v] = (float(str_list[1]), float(str_list[2]))
            else:
                vertices[v] = (int(str_list[1]), int(str_list[2]))
        i += 1

    f.close()
    return vertices


def read_file(file_path: str, key=' '):
    """Return as dictionary of lists
        data[line #] = [str] """

    f = open(file_path, 'r')

    data = {}

    i = 0
    for line in f.readlines():
        i += 1
        data[i] = read_line(line, key)

    f.close()
    return data


def manual_ss_edges(n=53):
    """School graph vertices connections (edges)
    edges = [(u, v), .... ]
    Note: index range 1,...n"""

    V = list(range(n))

    edges_ = []

    # cafeteria
    edges_ = adj_nodes(edges_, 0, 3)
    edges_ = adj_nodes(edges_, 3, 9)
    edges_.append((2, 5))
    edges_.append((0, 3))
    edges_.append((1, 4))

    # hall 3
    edges_.append((7, 9))
    edges_.append((8, 6))
    edges_ = adj_nodes(edges_, 9, 13)

    # room G
    edges_.append((13, 14))
    edges_.append((13, 15))
    edges_.append((14, 16))
    edges_.append((16, 15))
    edges_.append((15, 12))

    # hall 2
    edges_.append((12, 17))
    edges_.append((17, 18))
    edges_.append((23, 24))
    edges_.append((24, 29))
    edges_.append((24, 18))
    edges_.append((29, 30))

    # room F
    edges_.append((18, 21))
    edges_.append((22, 21))
    edges_.append((22, 20))
    edges_.append((19, 20))
    edges_.append((19, 21))

    # room E
    edges_.append((25, 26))
    edges_.append((27, 28))
    edges_.append((26, 28))
    edges_.append((25, 27))
    edges_.append((24, 27))

    # room D
    edges_.append((33, 30))
    edges_.append((34, 32))
    edges_.append((32, 31))
    edges_.append((33, 34))
    edges_.append((33, 31))

    # hall 1
    edges_.append((30, 35))
    edges_.append((35, 36))
    edges_.append((35, 37))
    edges_.append((38, 37))
    edges_.append((39, 37))
    edges_.append((39, 40))

    # gym
    edges_ = adj_nodes(edges_, 41, 45)
    edges_ = adj_nodes(edges_, 45, 49)
    edges_ = adj_nodes(edges_, 49, 53)
    edges_.append((41, 45))
    edges_.append((49, 45))
    edges_.append((42, 46))
    edges_.append((50, 46))
    edges_.append((51, 47))
    edges_.append((43, 47))
    edges_.append((44, 48))
    edges_.append((52, 48))
    edges_.append((52, 39))

    edges = []
    for el in edges_:
        v1 = el[0] + 1
        v2 = el[1] + 1
        edges.append((v1, v2))

    return edges, edges_


def adj_nodes(edges: list, s, e):

    i = s
    for v in range(s, e):
        j = i + 1
        if j < e:
            edges.append((i, j))
        i += 1

    return edges


def as_list(V_dict: dict):
    """Transform dictionary in list
    V = [(x1, y1), (x2, y2)...(xn, yn)]"""

    V = []

    for key in V_dict.keys():
        if isinstance(key, int):
            v_x = V_dict[key][0]
            v_y = V_dict[key][1]
            V.append((v_x, v_y))

    return V


def make_txt_edges():

    edges = manual_ss_edges()[0]

    line1 = ['SS-1', 'Edges']
    line2 = ['ID', 'v1', 'v2']
    line_f = '{0:3d} {1:2d} {2:2d} \n'

    make_txt('School_EGraph', edges, line1, line2, line_f)


def make_txt_pose():

    pose = extract_robot_motion()

    line1 = ['SS-1', 'Robot Pose']
    line2 = ['ID', 'x', 'y']
    line_f = '{0:3d} {1:8.4f} {2:8.4f} \n'

    make_txt('School_RPose', pose, line1, line2, line_f)


def make_txt(file_name: str, my_list: list, line1: list, line2: list, line_f='{0:3d} {1:8.4f} {2:8.4f} \n'):
    f_path = get_file_path(file_name)
    f = open(f_path, 'w+')

    line1 = '{0:4s} {1:8s} \n'.format(line1[0], line1[1])
    line2 = '{0:4s} {1:8s} {2:6s} \n'.format(line2[0], line2[1], line2[2])
    f.write(line1)
    f.write(line2)

    i = 1
    for el in my_list:

        a = el[0]
        b = el[1]

        my_text = line_f.format(i, a, b)
        f.write(my_text)
        i += 1

    f.close()

    return


def make_txt_str(file_name: str, my_list: list, line1: list, line2: list, line_f='{0:4s} {1:8s} {2:8s} {3:8s} {4:8s} \n'):

    f_path = get_file_path(file_name)
    f = open(f_path, 'w+')

    line1 = '{0:4s} {1:8s} \n'.format(line1[0], line1[1])
    line2 = '{0:4s} {1:8s} {2:8s} {3:8s} {4:8s} \n'.format(line2[0], line2[1], line2[2], line2[3], line2[4])
    f.write(line1)
    f.write(line2)

    for el in my_list:
        name = el[0]
        c1 = str(el[1])
        c2 = str(el[2])
        c3 = str(el[3])
        c4 = str(el[4])

        my_text = line_f.format(name, c1, c2, c3, c4)
        f.write(my_text)

    f.close()

    return


def extract_robot_motion(name_file='School_CamsLeft'):

    # find file
    f_path = get_file_path(name_file, 'txt_files', '.csv')
    data = read_file(f_path, ',')

    pose = []

    # delta = (48.15, 12.39)
    delta = (0, 0)
    # delta = (45.8, 34-18)

    for k in data.keys():
        if k == 1:
            continue
        line = data[k]
        x = float(line[1]) + delta[0]
        y = float(line[3]) + delta[1]
        z = float(line[2])

        if z > 2.0:
            break
        else:
            pose.append((x, y))

    return pose


def get_vertices(name_file, parent_folder='txt_files'):
    file_path = get_file_path(name_file, parent_folder)
    V_dict = read_data_txt(file_path)
    V = as_list(V_dict)

    return V


def get_edges(name_file, parent_folder='txt_files'):
    file_path = get_file_path(name_file, parent_folder)
    E_dict = read_data_txt(file_path)
    E = as_list(E_dict)

    return E


def get_info(name_file, what='V', parent_folder='txt_files'):
    """Get data from txt file and convert to data list
    :param name_file : name of the file, without txt extension
    :param what : V = vertices, E = edges, R = pose
    :param parent_folder"""
    file_path = get_file_path(name_file, parent_folder)

    if what == 'V' or what == 'R':
        my_type = 'float'
    else:
        my_type = 'int'

    data_dict = read_data_txt(file_path, my_type)
    data = as_list(data_dict)

    return data


def trans_pose(delta: tuple):
    """parse robots pose and translate according to delta
    save as txt file"""

    pose = get_info('School_RPose', 'R')

    # translate
    pose2 = []
    for pt in pose:
        x = (pt[0] + delta[0])
        y = (pt[1] + delta[1])
        pose2.append((x, y))

    return pose2


if __name__ == '__main__':
    # make_txt_edges()
    # make_txt_pose()
    trans_pose((0, 0))


