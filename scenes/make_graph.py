import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from math import sqrt
from milp_mespp.core import extract_info as ext
from milp_mespp.core import plot_fun as pf
from milp_mespp.core import create_parameters as cp

from milp_sim.scenes.class_room import MyRoom
from milp_sim.scenes import files_fun as ff

# ieee compliant plots
plt.rcParams.update({'text.usetex': True})
plt.rcParams.update({'font.family': 'serif'})


a_x = 1.1


# scenarios
def ss_1():
    """Parameters of the School Scenario
    MyRoom(label, dim, c0, door)
    label: str
    dim: (x, y)
    c0: (x0, y0) - bottom left
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


def ss_2(bloat=False, adjusted=True):
    """Parameters of the School Scenario [floor plan, no origin]
        MyRoom(label, dim, c0=None, door=None)
        label: str
        dim: (x, y)
        c0: (x0, y0) - bottom left
        door: (x0, y0, 'v/h', size)"""

    n_rooms = 12
    room_list = list(range(1, n_rooms + 1))

    dim = [(0, 0), (17, 31), (3.4, round(16.4 - 3.9, 2)), (5.83, 3), (5.83, 3), (5.83, 3),
           (7.9, 5.5), (7.9, 5.5), (7.9, 5.5), (7.9, 5.5), (10, 17), (60-10, 3.9), (10, 8.3)]

    if bloat:
        dim_b = [(a_x * el[0], a_x * el[1]) for el in dim]

        if adjusted:
            # manual corrections
            dim_b[6] = (a_x * (7.9 + 1.6), a_x * 5.5)
            dim_b[7] = (a_x * (7.9 + 1.6), a_x * 5.5)
            dim_b[8] = (a_x * (7.9 + 1.6), a_x * 5.5)
            dim_b[9] = (a_x * (7.9 + 1.6), a_x * 5.5)
            dim_b[10] = (a_x * (10 + 1.5), (a_x * 17) + 4.5)

        dim = [(round(el[0], 2), round(el[1], 2)) for el in dim_b]

    school = ext.create_dict(room_list, None)

    # Gym
    school[1] = MyRoom('Gym', dim[1])

    # Hall
    H1 = MyRoom('H1', dim[2])
    H2 = MyRoom('H2', dim[11])
    H3 = MyRoom('H3', dim[12])

    school[2] = H1
    school[11] = H2
    school[12] = H3

    # small rooms
    school[3] = MyRoom('A', dim[3])
    school[4] = MyRoom('B', dim[4])
    school[5] = MyRoom('C', dim[5])
    # classrooms
    school[6] = MyRoom('D', dim[6])
    school[7] = MyRoom('E', dim[7])
    school[8] = MyRoom('F', dim[8])
    school[9] = MyRoom('G', dim[9])
    school[10] = MyRoom('Cafe', dim[10])

    return school


# placing wrt origin
def delta_origin(bloat=False):
    """origin is at bottom left (0, 0) of floor plan
    (encounter of cafe vertical and gym horizontal)"""

    x_cafe = 10
    x_hall = 60
    x_gym = 17
    y_cafe = 1.8

    dx = x_cafe + x_hall - x_gym
    dy = 0

    return dx, dy


def space_between(bloat=False):
    """space between rooms of school"""

    # Gym-A, A-B, B-C
    y = [1.2, 0.6, 0.6]
    # H2-D, D-E, E-F, F-G
    x = [1.5, 1.4, 1.4, 1.4]

    if bloat:
        y_b = [a_x * el for el in y]
        x_b = [a_x * el for el in x]
        x_b[0] = 0
        x, y = x_b, y_b

    return x, y


def update_ref(room):
    x_right = room.c3[0]
    y_top = room.c3[1]

    x_left = room.c1[0]
    y_bottom = room.c1[1]

    return x_right, x_left, y_bottom, y_top


def place_ss2(school_g=ss_2(), bloat=False):
    """Place school rooms according to floorplan coordinates and delta origin"""

    # floorplan info
    dx, dy = delta_origin(bloat)
    spacex, spacey = space_between(bloat)

    dc = 1.8
    if bloat:
        dc = dc * a_x

    gym = school_g[1]
    h1 = school_g[2]
    h2 = school_g[11]
    h3 = school_g[12]
    a, b, c = school_g[3], school_g[4], school_g[5]
    d, e, f, g = school_g[6], school_g[7], school_g[8], school_g[9]
    cafe = school_g[10]

    gym.set_coordinates((dx, dy))

    # nivel da gym
    x_right, x_left, y_bottom, y_top = update_ref(gym)

    # hall 1
    h1_c0 = (x_right - h1.dim[0], y_top)
    h1.set_coordinates(h1_c0)

    # A
    c0 = (x_right-h1.dim[0]-a.dim[0], y_top + spacey[0])
    a.set_coordinates(c0)
    x_right, x_left, y_bottom, y_top = update_ref(a)

    # b
    c0 = (x_left, y_top + spacey[1])
    b.set_coordinates(c0)
    x_right, x_left, y_bottom, y_top = update_ref(b)
    # c
    c0 = (x_left, y_top + spacey[2])
    c.set_coordinates(c0)
    x_right, x_left, y_bottom, y_top = update_ref(h1)

    # hall 2
    c0 = (x_right-h2.dim[0], y_top)
    h2.set_coordinates(c0)
    x_right, x_left, y_bottom, y_top = update_ref(h2)

    # D
    c0 = (x_right - spacex[0] - d.dim[0], y_top)
    d.set_coordinates(c0)
    x_right, x_left, y_bottom, y_top = update_ref(d)

    # E
    c0 = (x_left - spacex[1] - e.dim[0], y_bottom)
    e.set_coordinates(c0)
    x_right, x_left, y_bottom, y_top = update_ref(e)

    # F
    c0 = (x_left - spacex[2] - f.dim[0], y_bottom)
    f.set_coordinates(c0)
    x_right, x_left, y_bottom, y_top = update_ref(f)

    # G
    c0 = (x_left - spacex[3] - g.dim[0], y_bottom)
    g.set_coordinates(c0)
    x_right, x_left, y_bottom, y_top = update_ref(h2)

    # hall 3
    c0 = (x_left-h3.dim[0], y_bottom)
    h3.set_coordinates(c0)
    x_right, x_left, y_bottom, y_top = update_ref(h3)

    # cafe
    c0 = (x_left-cafe.dim[0], y_top-dc-cafe.dim[1])
    cafe.set_coordinates(c0)
    x_right, x_left, y_bottom, y_top = update_ref(cafe)

    # save
    school = dict()
    school[1] = gym
    school[2] = h1
    school[3] = a
    school[4] = b
    school[5] = c
    school[6] = d
    school[7] = e
    school[8] = f
    school[9] = g
    school[10] = cafe
    school[11] = h2
    school[12] = h3

    return school


def save_coord_ss2(bloat=False):
    """Create txt and save floor plan coordinates
    :param bloat : True for original size (1:1), False for Gazebo size (1:1.1)
    """

    # floorplan
    school_1 = ss_2(bloat)
    # place in space (delta origin)
    school = place_ss2(school_1, bloat)

    coord_list = []
    for k in school.keys():
        room = school[k]
        el = [room.label, room.c1, room.c2, room.c3, room.c4, room.dim]

        coord_list.append(el)

    if bloat:
        line1 = ["SS-2", "Floorplan Coordinates - Images"]
        f_name = "School_Image_Coordinates"
    else:
        line1 = ["SS-2", "Floorplan Coordinates - Gazebo"]
        f_name = "School_Gazebo_Coordinates"

    line2 = ["ID", "C1", "C2", "C3", "C4", "DIM"]

    ff.make_txt_str(f_name, coord_list, line1, line2)


# building scenario
def build_school(bloat=False, adjusted=False):
    # floorplan
    school_g = ss_2(bloat, adjusted)
    # place in space (delta origin)
    school_block = place_ss2(school_g, bloat)
    # merge halls for nice plotting
    school = merge_hall(school_block)

    return school


def merge_hall(school_block):
    """Merge H1, H2, H3 for nice plotting"""

    school = dict()
    h1 = school_block[2]
    h2 = school_block[11]
    h3 = school_block[12]

    h1.merge(h2.dim, h2.c1)
    h1.merge(h3.dim, h3.c1)

    for room_id in school_block.keys():
        if room_id == 2:
            my_room = h1
        elif room_id > 10:
            break
        else:
            my_room = school_block[room_id]

        school[room_id] = my_room

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


def wall_nodes_connections(scenario: dict):
    """Loop through scenario and add nodes and connections to lists
    :param scenario : dictionary of rooms (class)
    return :
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


# build school for gazebo
def build_gazebo_ss2():
    """Shrink graph for images to fit gazebo world
    """

    # read from final images graph (already organized)
    V_i = ff.get_vertices('School_Image_VGraph')
    # get edges
    E_i = ff.get_edges('School_Image_EGraph')

    V = assign_coordinates(V_i)

    # make txt - vertices
    f_name = 'School_Gazebo_VGraph'
    line1 = ['SS-2', 'Gazebo Graph Coordinates']
    line2 = ['ID', 'X', 'Y']

    ff.make_txt(f_name, V, line1, line2)

    # make txt - edges
    f_name_e = 'School_Gazebo_EGraph'
    # edges will be the same
    E = E_i
    ff.make_txt_edges(f_name_e, E)

    # plot to make sure it looks right (school + graph)
    plot_all()

    # make igraph file


def shrink_vertices(V_coord: list, E: list):
    """fitting: a_x = 1.1
    shrink by distance """


    n = len(V_coord)

    V = list(range(1, n + 1))

    dist_i = []

    E.sort(key=lambda k: [k[0], k[1]])

    new_dist = []
    for edge in E:
        v1 = edge[0]
        v2 = edge[1]

        # coordinates
        p1 = V_coord[v1-1]
        p2 = V_coord[v2-1]

        # get original distance
        dx = - p2[0] + p1[0]
        dy = - p2[1] + p1[1]
        dist_i.append((dx, dy))

        # shrink it
        dx_new = shrink_it(dx)
        dy_new = shrink_it(dy)
        d_new = (dx_new, dy_new)

        new_dist.append(d_new)

    # first vertex
    xo, yo = V_coord[0][0], V_coord[0][1]
    Vxy_new = [(xo, shrink_it(yo))]
    V_new = [1]

    E_aux = [edge for edge in E]

    while True:
        for edge in E_aux:
            v1 = edge[0]
            v2 = edge[1]

            i = E.index(edge)
            j = E_aux.index(edge)

            if v1 in V_new and v2 not in V_new:
                my_v = v1
            elif v2 in V_new and v1 not in V_new:
                my_v = v2
            elif v1 in V_new and v2 in V_new:
                E_aux.pop(j)
            else:
                pass

            xo = Vxy_new[my_v - 1][0]
            yo = Vxy_new[my_v - 1][1]

            x1 = round(xo + new_dist[i][0], 2)
            y1 = round(yo + new_dist[i][1], 2)

            V_new.append(v1)
            Vxy_new.append((x1, y1))

            E_aux.pop(j)
            break

        if len(V_new) == len(V):
            break

    return Vxy_new


def assign_coordinates(V_xy: list):

    n = len(V_xy)
    V = list(range(1, n + 1))

    xo = V_xy[0][0]
    yo = shrink_it(V_xy[0][1])

    x, y = xo, yo

    V_xy_new = [(xo, yo)]

    for v in V:
        vidx = v - 1

        if v == 1:
            continue

        # coordinates
        p1 = V_xy[vidx - 1]
        p2 = V_xy[vidx]

        # get original distance
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]

        # shrink it
        dx_new = shrink_it(dx)
        dy_new = shrink_it(dy)

        if v in [16]:
            dx_new = shrink_it(dx, 1 / 1.4)

        if v in [19]:
            dx_new = shrink_it(dx, 1 / 1.6)

        if v in [22, 28, 34]:
            # x += 0.8
            dx_new = shrink_it(dx, 1/1.4)

        if v == 31:
            dx_new = shrink_it(dx, 1/1.3)

        if v == 13:
            dx_new -= 1.5

        if v in [37, 38, 39]:
            dx_new = shrink_it(dx, 1.3)

        if v > 40:
            dy_new = shrink_it(dy, 1/1.4)
            dx_new = shrink_it(dx, 1/1.2)

        # new pose
        x = xo + dx_new

        if v in [19]:
            x -= 1

        y = yo + dy_new

        V_xy_new.append((x, y))

        # iterate
        xo = x
        yo = y

    return V_xy_new


def shrink_it(it, alpha=1/a_x):
    b = round(it * alpha, 2)
    return b


def get_dist(p1, p2):
    """Get distance between points"""
    x1 = p1[0]
    x2 = p2[0]

    y1 = p1[1]
    y2 = p2[1]

    dist = sqrt((x1-x2) ** 2 + (y1-y2) ** 2)

    return dist


# build actual g file
def create_igraph():

    edges = ff.get_edges('School_Gazebo_EGraph')
    g = cp.create_school_graph(edges)
    return

# ----------------------------------------------------------------------------------
# PLOT STUFF

def save_plot(fig_name: str, folder='figs', my_ext='.png'):

    fig_path = ff.get_file_path(fig_name, folder, my_ext)

    plt.xlabel('X [m]')
    plt.ylabel('Y [m]')

    plt.savefig(fig_path, facecolor=None, edgecolor=None,
                orientation='landscape', transparent=True)

    plt.show()


def finish_plot(op='all', lgd=None, my_ext='.png'):
    """Add title and legend
    Save plot as png"""

    fig_name = ''
    folder = 'figs'

    if 'all' in op:
        if 'gazebo' in op:
            plt.title('Gazebo School Scenario')
            fig_name = 'school_gazebo'
        else:
            plt.title('School Scenario')
            fig_name = 'school_graph_final'
        if lgd is not None:
            my_hand = []
            if len(lgd) >= 2:
                floorplan = mlines.Line2D([], [], color='k', label=lgd[0])
                graph = mlines.Line2D([], [], color='m', label=lgd[1], marker='o', markersize=5)
                my_hand = [floorplan, graph]
            if len(lgd) == 3:
                robot = mlines.Line2D([], [], color='b', label=lgd[2])
                my_hand.append(robot)
            plt.legend(handles=my_hand)

    elif op == 'pose':
        plt.title('Robot Pose \n(non-aligned)')
        fig_name = 'robotpose_1'
        if lgd is not None:
            robot = mlines.Line2D([], [], color='b', label=lgd[0])
            plt.legend(handles=[robot])

    elif op == 'school':
        plt.title('School Floorplan')
        fig_name = 'floorplan'

    elif op == 'graph':
        plt.title('School Graph')
        fig_name = 'graph'

    elif op == 'both':
        plt.title('School Floorplans')
        fig_name = 'bloated_school'
        if lgd is not None:
            gazebo = mlines.Line2D([], [], color='teal', label=lgd[0])
            bloat = mlines.Line2D([], [], color='silver', label=lgd[1])
            adjusted = mlines.Line2D([], [], color='k', label=lgd[2])
            plt.legend(handles=[gazebo, bloat, adjusted])

    elif op == 'mis':
        plt.title('School Positioning')
        fig_name = 'school_aligning'
        if lgd is not None:
            origin = mlines.Line2D([], [], color='gray', label=lgd[0])
            school = mlines.Line2D([], [], color='k', label=lgd[1])
            robot = mlines.Line2D([], [], color='b', label=lgd[2])
            robot0 = mlines.Line2D([], [], color='c', label=lgd[4])
            pt = mlines.Line2D([], [], color='r', label=lgd[3])
            plt.legend(handles=[origin, school, robot, robot0, pt], loc=3)

    elif op == 'align':
        plt.title('School Floorplan\n[from robot pose 1:1]')
        fig_name = 'school_aligned'
        if lgd is not None:
            school = mlines.Line2D([], [], color='k', label=lgd[0])
            robot = mlines.Line2D([], [], color='b', label=lgd[1])
            plt.legend(handles=[school, robot])

    elif op == 'pose_school':
        plt.title('School Floorplan')
        fig_name = 'school_pose'
        school = mlines.Line2D([], [], color='k', label=lgd[0])
        robot = mlines.Line2D([], [], color='b', label=lgd[1])
        plt.legend(handles=[school, robot])

    save_plot(fig_name, folder, my_ext)


def plot_all(bloat=False):
    """plot school scenario from DISC"""

    # SCHOOL
    plot_ss2(bloat)

    # GRAPH
    numbers = True
    edges = True
    plot_graph(bloat, numbers, edges)

    lgd = ['Floorplan', 'Graph']

    if bloat:
        # POSE
        f_name = 'School_RPose_Translated'
        plot_pose(f_name)
        lgd.append('Robot Pose')
        op = 'all'
    else:
        op = 'all_gazebo'

    finish_plot(op, lgd)


def plot_pose(f_name='School_RPose', finish=False, color='b-'):
    """Plot robot pose as extracted from csv file"""

    if isinstance(f_name, str):
        # get pose info
        RPose = ff.get_info(f_name, 'R')
    else:
        RPose = f_name

    R_x = [el[0] for el in RPose]
    R_y = [el[1] for el in RPose]
    plt.plot(R_x, R_y, color)

    if finish:
        finish_plot("pose", ['Robot'])


def plot_graph(bloat=False, number=False, edges=True, finish=False):
    """Plot graph (not numbered)"""

    # get graph info
    if bloat:
        f_name = ['School_Image_VGraph', 'School_Image_EGraph']
    else:
        f_name = ['School_Gazebo_VGraph', 'School_Gazebo_EGraph']

    # retrieve vertices and edges
    V = ff.get_info(f_name[0], 'V')
    E = ff.get_info(f_name[1], 'E')

    my_color = 'm'

    if edges:
        # organize edges as list with python index (0...n-1)
        E_plot = []
        for el in E:
            v1 = el[0] - 1
            v2 = el[1] - 1
            E_plot.append((v1, v2))
        pf.plot_points_between_list(V, E_plot, my_color, 'o')
    else:
        pf.plot_points(V, my_color, 'o', 2)

    if number:
        # plot index + 1 at v coordinates +- offset
        for coord in V:
            i = V.index(coord) + 1
            x = coord[0] + 0.5
            y = coord[1] + 0.75
            plt.text(x, y, str(i), fontsize=8, color=my_color)

    if finish:

        finish_plot('graph')


def plot_ss2(bloat=False, adjusted=True, my_color='k'):
    """Plot school scenario (floorplan only)"""
    # set up scenario
    school = build_school(bloat, adjusted)
    # school floor plan
    wall_nodes, wall_conn = wall_nodes_connections(school)

    # plot floor plan
    pf.plot_points_between_list(wall_nodes, wall_conn, my_color)

    return school


def plot_mismatch():
    # gazebo
    plot_ss2()
    # translated pose
    f_name = 'School_RPose_Translated'
    plot_pose(f_name)

    op = 'pose_school'
    lgd = ['Gazebo', 'Robot']
    finish_plot(op, lgd)


def align_pose_school():

    # frame details - robot entering gym
    # visual: 1032, (4.85, -15.29)
    pose_raw = ff.get_info('School_RPose', 'R')
    frame_number = 1040
    xy_frame = pose_raw[frame_number]

    # SCHOOL
    bloat = True
    school = plot_ss2(bloat)
    # floorplan detail - gym door
    xy_door = (school[1].c1[0], school[1].c1[1] + a_x * (2 + 0.9) + 0.5)
    # and robot POSE
    plot_pose(pose_raw, False, 'c')
    # plot intersection points
    pts = [xy_frame, xy_door]
    pf.plot_points_between_list(pts, [(0, 1)], 'r', 'o')

    # translate
    delta = (round(xy_door[0] - xy_frame[0], 2), round(xy_door[1] - xy_frame[1], 2))
    print(delta)
    pose2 = ff.trans_pose(delta)
    # save pose translated
    ff.make_txt('School_RPose_Translated', pose2, ['SS-2', 'Robot Pose Translated'], ["ID", 'x', 'y'])

    origin = [(0, 0), (-50, 0), (60, 0), (0, -20), (0, 60)]
    or_conn = [(0, 1), (0, 2), (0, 3), (0, 4)]

    plot_pose(pose2, False, 'blue')
    pf.plot_points_between_list(origin, or_conn, 'gray')

    # plot_graph()
    lgd = ['Origin', 'School', 'Robot', 'Matching Point', 'Robot (raw)']
    finish_plot('mis', lgd)


def plot_both_ss2():

    plot_ss2(False, False, 'teal')

    plot_ss2(True, False, 'silver')

    plot_ss2(True, True, 'black')

    finish_plot('both', ['Gazebo', 'Bloated', 'Adjusted'])


def compute_for_mesh():

    bloat = True
    school_ = ss_2(bloat)

    school = place_ss2(school_, bloat)

    dc = 1.8
    if bloat:
        dc = dc * a_x

    gym = school[1]
    h1 = school[2]
    h2 = school[11]
    h3 = school[12]
    a, b, c = school[3], school[4], school[5]
    d, e, f, g = school[6], school[7], school[8], school[9]
    cafe = school[10]

    print('xo, yo = %s' % str(cafe.c1))

    print('cafe dimensions: %s' % str(cafe.dim))
    print('Hall 3 dimensions: %s' % str(h3.dim))
    # slide h3 wrt cafe
    dy_up = round(cafe.dim[1] + dc - h3.dim[1], 2)
    door_cafe = round(h2.dim[1]/2 + 0.25, 2)
    print('Attach cafe-h3: %s, %s' % (str(dy_up), str(door_cafe)))

    # stairs
    stairs = round(abs(h3.c4[0]-g.c1[0]), 2)
    print('H2.S dimensions (%s, %s)' % (str(stairs), str(h2.dim[1])))
    print('H2.S attach (%s, %s) \n' % (str(0.0), str(h2.dim[1]/2)))

    # HALL classrooms G --> F
    h2G_x = round(f.c1[0]-g.c1[0], 2)
    h2G_y = h2.dim[1]
    print('H2.G dimensions (%s, %s)' % (str(h2G_x), str(h2G_y)))

    # classrooms F --> E
    h2F_x = round(e.c1[0]-f.c1[0], 2)
    h2F_y = h2.dim[1]
    print('H2.F dimensions (%s, %s)' % (str(h2F_x), str(h2F_y)))

    # classrooms E --> D
    h2E_x = round(d.c1[0]-e.c1[0], 2)
    h2E_y = h2.dim[1]
    print('H2.E dimensions (%s, %s)' % (str(h2E_x), str(h2E_y)))

    # classrooms D --> END
    h2D_x = round(h2.c3[0] - d.c1[0], 2)
    h2D_y = h2.dim[1]
    print('H2.D dimensions (%s, %s)' % (str(h2D_x), str(h2D_y)))

    # attach classrooms
    print('Attach classrooms: (0, %s)' % str(h2D_y/2))
    h2G_door = round(g.dim[0] - 0.8 - (0.9/2), 2)
    h2G_D = d.dim
    print('G-D rooms dimensions: %s' % str(h2G_D))
    print('G-D attach doors (0.0, %s) \n' % str(h2G_door))

    # H1
    h1_h2D = round(h2D_x - h1.dim[0], 2)
    print('Attach H1-H2 (%s, %s)' % (str(h1_h2D), str(h1.dim[0]/2)))

    # room C
    h1C_x = h1.dim[0]
    h1C_y = round(h1.c2[1]-c.c1[1], 2)
    print('H1C dimensions (%s, %s)' % (str(h1C_x), str(h1C_y)))

    # room B
    h1B_x = h1.dim[0]
    h1B_y = round(c.c1[1] - b.c1[1], 2)
    print('H1B dimensions (%s, %s)' % (str(h1B_x), str(h1B_y)))

    # room A
    h1A_x = h1.dim[0]
    h1A_y = round(b.c1[1] - gym.c2[1], 2)
    print('H1A dimensions (%s, %s)' % (str(h1A_x), str(h1A_y)))
    h1A_att = round(a.c1[1]-gym.c2[1], 2)
    ABC_door = c.dim[1] - 1.05 - (0.9 / 2)

    # rooms A-B-C
    print('\nRooms A-B-C dimensions %s' % str(b.dim))
    print('H1B-C attach (0.0, %s)' % str(ABC_door))
    print('H1-A attach (%s, %s)' % (str(h1A_att), str(ABC_door)))

    # gym
    print('\nGym dimensions %s' % str(gym.dim))
    gym_x = round(h1.dim[0] - gym.dim[0], 2)
    gym_d = round(gym.dim[0] - (h1.dim[0]/2), 2) # + 0.5
    print('Attach Gym (%s, %s)' % (str(gym_x), str(gym_d)))

    plot_all(bloat)


if __name__ == '__main__':
    create_igraph()
    # build_gazebo_ss2()
    # bloat = True
    # plot_all(bloat)
    # plot_bloat_school_pose()
    # align_pose_school()
    # plot_both_ss2()
    # save_coord_ss2(True)
    #
    # compute_for_mesh()
    # align_pose_school()
    # plot_mismatch()
    # plot_both_ss2()

    # save_coord_ss2(True)
