import matplotlib.pyplot as plt
from class_room import MyRoom
import files_fun as ff
from milp_mespp.core import extract_info as ext
import matplotlib.patches as mpatches
import matplotlib.lines as mlines


# scenarios
def ss_1():
    """Parameters of the School Scenario
    MyRoom(label, dim, c0, door)
    label: str
    dim: (x, y)
    c0: (x0, y0)
    door: (x0, y0, 'v/h', size)"""

    n_rooms = 10
    n_doors = 10

    r_list = list(range(1, n_rooms + 1))
    d_list = list(range(1, n_doors + 1))

    school = ext.create_dict(r_list, None)
    doors = ext.create_dict(d_list, None)

    # Gym
    school[1] = MyRoom('Gym', (17, 31), (53, 0))

    doors[1] = {'xy': (67.4, 31.), 'o': 0, 'size': 0.9}
    # school[1].place_door(doors[1])

    # Hall
    H1 = MyRoom('H', (3.5, 12.4), (66.5, 31))
    H1.merge((60, 4), (10, 43.4))
    H1.merge((10, 4.3), (10, 47.4))
    school[2] = H1

    # small rooms
    school[3] = MyRoom('A', (5.8, 3), (60.7, 32.2))
    doors[3] = {'xy': (66.5, 33), 'o': 1, 'size': 0.9}

    school[4] = MyRoom('B', (5.8, 3), (60.7, 35.8))
    doors[4] = {'xy': (66.5, 36.6), 'o': 1, 'size': 0.9}

    school[5] = MyRoom('C', (5.8, 3), (60.7, 39.4))
    doors[5] = {'xy': (66.5, 40.2), 'o': 1, 'size': 0.9}

    school[6] = MyRoom('D', (8, 5.5), (60.5, 47.4))
    doors[6] = {'xy': (66.8, 47.4), 'o': 0, 'size': 0.9}

    school[7] = MyRoom('E', (8, 5.5), (51.1, 47.4))
    doors[7] = {'xy': (57.4, 47.4), 'o': 0, 'size': 0.9}

    school[8] = MyRoom('F', (8, 5.5), (41.7, 47.4))
    doors[8] = {'xy': (48.0, 47.4), 'o': 0, 'size': 0.9}

    school[9] = MyRoom('G', (8, 5.5), (32.3, 47.4))
    doors[9] = {'xy': (38.6, 47.4), 'o': 0, 'size': 0.9}

    school[10] = MyRoom('Cafe', (10, 17), (0, 33.5))
    doors[10] = {'xy': (10, 46.7), 'o': 1, 'size': 0.9}

    return school


def house_hri():
    """label: str
    dim: (x, y)  c0: (x0, y0)"""
    n_rooms = 24
    r_list = list(range(1, n_rooms + 1))
    house = ext.create_dict(r_list, None)

    house[1] = MyRoom('Garage', (5.1, 5.8), (1.7, -10.4))
    house[2] = MyRoom('Lounge', (3.8, 3.7), (1.9, -4.4))
    house[3] = MyRoom('Living', (3.8, 5.2), (2.0, -0.5))
    house[4] = MyRoom('Dining', (2.7, 6.6), (3.1, 4.9))
    house[5] = MyRoom('Porch', (1.5, 1.7), (-0.1, -11.7))
    house[6] = MyRoom('Coat', (1.7, 2.9), (-2.1, -10.9))
    house[7] = MyRoom('WC1', (1.1, 0.8), (-1.1, -6))
    house[8] = MyRoom('PDR', (1.9, 1.2), (-1.8, -5))
    house[9] = MyRoom('Pantery', (1.6, 2.0), (-2.2, -2.3))
    house[10] = MyRoom('Rumpus', (4, 3.9), (-1.1, 7.7))
    house[11] = MyRoom('Master BD', (4.1, 4), (-6.5, -10.2))
    house[12] = MyRoom('Study', (4.1, 2.8), (-6.5, -3.7))
    house[13] = MyRoom('BD2', (4, 2.9), (-6.5, -0.7))
    house[14] = MyRoom('BD3', (3.2, 3.4), (-6.5, 2.4))
    house[15] = MyRoom('Bath 1', (2.9, 1.9), (-6.5, 6.0))
    house[16] = MyRoom('BD4', (3.1, 3.4), (-6.5, 8.1))
    house[17] = MyRoom('Laundry', (1.9, 2.2) , (-3.2, 9.3))
    house[18] = MyRoom('Bath2', (1.4, 0.9), (-2.7, -6))
    house[19] = MyRoom('Kitchen', (3.9, 7.6), (-2.2, -0.1))
    house[20] = MyRoom('Master Bath', (3.6, 2.1), (-6.5, -6.0))
    # halls
    house[21] = MyRoom('Hall 1', (1.5, 3.5), (-0.1, -9.8))
    house[22] = MyRoom('Hall 2', (1.3, 5.5), (0.3, -5.8))
    house[23] = MyRoom('Hall 3', (1, 2.3), (1.9, 5.1))
    house[24] = MyRoom('Hall 4', (0.8, 6.0), (-3.2, 2.5))

    return house


# generic functions
def nodes_and_conections(scenario: dict):
    """Loop through scenario and add nodes and connections to lists
    wall_nodes = [(x1, y1), (x2, y2)...]
    wall_conn = [(p1, p2)..]"""

    wall_nodes = []
    wall_conn = []

    for k in scenario.keys():

        r = scenario[k]
        # keep track of existent nodes
        i = len(wall_nodes)

        # get corner points from each room
        p = r.c

        # append to list
        wall_nodes = wall_nodes + p

        n_c = len(p) - 1
        # walls: 0-1, 1-2, 2-3, 3-1
        for j in range(n_c):
            wall_conn.append((i+j, i+j+1))
        wall_conn.append((i, i+n_c))

    return wall_nodes, wall_conn


def plot_points_between_list(v_points, v_conn, color, style='.-'):

    my_handle = None
    for wall_k in v_conn:
        i0 = wall_k[0]
        i1 = wall_k[1]

        n0 = v_points[i0]
        n1 = v_points[i1]

        px = [n0[0], n1[0]]
        py = [n0[1], n1[1]]

        my_handle = plt.plot(px, py, color + style)

    return my_handle


def plot_points(vertices: dict, color, style='o'):
    """Plot vertices from V dictionary"""
    my_handle = None

    for k in vertices.keys():
        if not isinstance(k, int):
            continue
        x = [vertices[k][0]]
        y = [vertices[k][1]]

        my_handle = plt.plot(x, y, color + style,  markersize=2)

    return my_handle


def save_plot(fig_name: str, folder='figs', my_ext='.pdf'):

    fig_path = ff.get_file_path(fig_name, folder, my_ext)

    plt.xlabel('X [m]')
    plt.ylabel('Y [m]')

    plt.savefig(fig_path, facecolor=None, edgecolor=None,
                orientation='landscape', transparent=True)

    plt.show()


def finish_plot(op='school', lgd=None):

    fig_name = ''
    my_ext = '.pdf'
    folder = 'figs'

    if op == 'school':
        plt.title('School Scenario')
        fig_name = 'school'
        if lgd is not None:
            floorplan = mlines.Line2D([], [], color='k', label=lgd[0])
            graph = mlines.Line2D([], [], color='m', label=lgd[1], marker='o', markersize=5)
            plt.legend(handles=[floorplan, graph])

    save_plot(fig_name, folder, my_ext)


def plot_school():
    """plot school scenario from DISC"""

    # set up scenario
    school = ss_1()

    # school floorplan
    wall_nodes, wall_conn = nodes_and_conections(school)
    # plot
    plot_points_between_list(wall_nodes, wall_conn, 'k', '-')

    # get graph info
    f_name = 'SchoolGraph'
    V = ff.get_vertices(f_name)
    # plot vertices
    graph = plot_points(V, 'm')

    lgd = ['Floorplan', 'Graph']

    finish_plot('school', lgd)


if __name__ == '__main__':
    plot_school()
