"""Visual example between explicit and implicit coordination algorithms"""
from milp_mespp.core import create_parameters as cp
from milp_mespp.classes.inputs import MyInputs
from milp_mespp.core import plan_fun as pln
from milp_mespp.core import extract_info as ext
from milp_sim.risk.src import r_plotfun as rpf
from matplotlib import pyplot as plt
import random
from math import sqrt


t0 = [8, 10, 14]

def choose_parameters(solver_type='central'):
    """Set simulation parameters"""
    specs = MyInputs()
    # searchers: default capture range = 0 (single vertex), zeta = 0 (no false negatives)
    # team size
    m = 3
    # planning horizon
    h = 3
    # graph: 4 x 4 grid
    graph = cp.create_grid_graph(4, 4)

    # target: default
    vo = [2, 5, 3]

    # set it up
    specs.set_searcher_together(False)
    specs.set_start_searchers(vo)
    specs.set_size_team(m)
    specs.set_all_times(h)
    specs.set_graph(graph)
    specs.set_start_target(t0)
    specs.set_target_motion('static')
    specs.set_solver_type(solver_type)

    return specs


def create_graph():
    g = cp.create_grid_graph(10, 10)

    vertices, edges = set_location(g)

    plot_graph(vertices, edges, 'k')

    plt.axis('off')

    fig_name = 'grid100_V_black'
    name_folder = 'coord'
    parent_folder = 'data'
    my_ext = '.pdf'

    save_plot(fig_name, name_folder, parent_folder, my_ext)


def plan_example(specs):
    """Implicit and explicit coordination algorithm"""

    output_data = True
    path_list, solver_data = pln.run_planner(specs, output_data)

    # pf.plot_plan_results(solver_data, name_folder, specs.start_target_true)

    return path_list, solver_data


def my_script():
    # set parameters
    specs_c = choose_parameters()
    specs_d = choose_parameters('distributed')
    g = specs_c.graph

    # create_folder
    folder_name = 'coord_simple23'
    folder_created = ext.folder_in_project('data/' + folder_name)

    # explicit coordination, get path
    c_path = plan_example(specs_c)[0]
    # implicit coordination, get path
    d_path = plan_example(specs_d)[0]

    c_search, d_search = search_path(c_path, d_path)
    plot_coordination_simple(g, c_search, d_search, folder_name)

    # illustrate search
    #
    # plot
    # plot_coordination(g, c_search, d_search)


def pick_next_v(g, vo):
    vo_idx = ext.get_python_idx(vo)
    # find neighbors
    vo_nei = [u_idx + 1 for u_idx in g.vs[vo_idx]["neighbors"]]

    # pick one random
    v1 = int(random.choice(vo_nei))

    return v1


def search_path(c_path, d_path):

    specs = choose_parameters()
    g = specs.graph
    h = specs.horizon

    m = specs.size_team
    S = ext.get_set_searchers(m)[0]

    # compute number of keys
    alpha = 3
    max_frame = m * alpha

    c_search = dict()
    d_search = dict()

    # starting point
    vo = {s: c_path[s][0] for s in S}
    c_search[0] = {s: [vo[s]] for s in S}
    d_search[0] = {s: [vo[s]] for s in S}

    # explicit
    for frame in range(1, max_frame + 1):
        # frame number
        c_search[frame] = dict()
        # each searcher is a new dict
        my_path = dict()
        for s in S:
            v1 = vo[s]
            # append first
            my_path[s] = [v1]
            # with h entries (one per step)
            for t in range(0, h):
                v1 = pick_next_v(g, v1)
                my_path[s].append(v1)

        c_search[frame] = my_path

    # implicit
    s_main = 1
    frame = 1
    while not s_main > S[-1]:

        # loop through searchers
        for j in range(3):

            # my_path[s] = [v1,...vh]
            my_path = dict()

            for s in S:
                # starting point
                v1 = vo[s]
                my_path[s] = [v1]

                # loop through steps
                for t in range(0, h):

                    if s == s_main:
                        if j == 0:
                            v1 = vo[s]
                        elif j == 1:
                            v1 = pick_next_v(g, v1)
                        else:
                            v1 = d_path[s][t + 1]
                    elif s < s_main:
                        v1 = d_path[s][t + 1]
                    else:
                        v1 = vo[s]

                    my_path[s].append(v1)

            # end of horizon
            d_search[frame] = my_path
            frame += 1
        s_main += 1

    #
    #
    #
    # for frame in range(1, max_frame + 1):
    #
    #     if frame % alpha == 0:
    #         s_main += 1
    #
    #     d_search[frame] = dict()
    #
    #     # each searcher is a new dict
    #     my_path_d = dict()
    #
    #     for s in S:
    #
    #         for j in range(3):
    #             # starting point
    #             v1 = vo[s]
    #             my_path_d[s] = [v1]
    #
    #             # with h entries (one per step)
    #             for t in range(0, h):
    #
    #                 if s == s_main:
    #                     if j == 0:
    #                         v1 = vo[s]
    #                     elif j == 1:
    #                         v1 = pick_next_v(g, v1)
    #                     else:
    #                         v1 = d_path[s][t + 1]
    #                 elif s < s_main:
    #                     v1 = d_path[s][t+1]
    #                 else:
    #                     v1 = vo[s]
    #
    #             my_path_d[s].append(v1)
    #
    #         d_search[frame] = my_path_d

    # final path

    # final path
    d_search[max_frame + 1] = c_path
    c_search[max_frame + 1] = d_path

    return c_search, d_search


def plot_custom_list(points, conn, settings):
    # unpack settings
    l_style = settings.line_style
    l_width = settings.line_width
    l_color = settings.line_color
    m_color = settings.marker_color
    m_size = settings.marker_size
    m_type = settings.marker

    if len(conn) < 1:
        px = [points[0]]
        py = [points[1]]

        plt.plot(px, py, color=l_color, linestyle=l_style, linewidth=l_width,
                 marker=m_type, markersize=m_size)

        plt.axis('off')

        return

    my_handle = None
    for edge in conn:
        i0 = edge[0]
        i1 = edge[1]

        n0 = points[i0]
        n1 = points[i1]

        px = [n0[0], n1[0]]
        py = [n0[1], n1[1]]

        my_handle = plt.plot(px, py, color=l_color, linestyle=l_style, linewidth=l_width,
                             marker=m_type, markersize=m_size)

        plt.axis('off')


def set_location(g, w=None):
    """Set (x,y) location for each vertex in graph"""

    V, n = ext.get_set_vertices(g)

    # vertices per line
    if w is None:
        w = int(sqrt(n))

    dx, dy = 2, 2
    xo, yo = 0.0, 0.0
    x, y = xo, yo

    counter = 0
    vertices = []
    edges = []

    # assign (x,y) location to each vertex
    for v in V:
        v_idx = ext.get_python_idx(v)

        if v == 1:
            # initial point
            x, y = xo, yo
        elif (v - 1) % w == 0:
            # new line
            x = xo
            y += dy
            # counter = 0
        else:
            # keep y, increase x
            x += dx
            y = y
            # counter += 1

        vertices.append((x, y))

        # save edges - (v1, v2)
        v_nei = [(v, u_idx + 1) for u_idx in g.vs[v_idx]["neighbors"]]
        for el in v_nei:
            if el not in edges:
                edges.append(el)

    return vertices, edges


def plot_path(vertices, edges, pi, settings):
    """Plot graph with x, y coordinate"""

    plot_graph(vertices, edges)

    for s in pi.keys():
        path = pi[s]
        s_edges = ext.path_conn(path)
        plot_custom_list(vertices, s_edges, settings)

    # plt.show()


def set_settings(m=3):
    my_colors = ['m', 'b', 'g', 'c']

    plot_settings = dict()

    for s in range(1, m + 1):
        plot_settings[s] = rpf.CustomizePlot()
        plot_settings[s].set_marker('o')
        # plot_settings[s].set_marker_size(5)

        # different color for each searcher
        plot_settings[s].set_l_color(my_colors[s-1])

        # thicker line
        plot_settings[s].set_l_width(5)
        plot_settings[s].set_marker_size(15)

    text_settings = rpf.CustomizePlot()

    my_text = dict()
    my_text[0] = 'Multi-Robot Search'
    my_text[1] = 'Team Coordination'
    my_text[2] = 'test2'
    my_text[3] = 'Explicit'
    my_text[4] = 'Implicit'

    text_settings.set_text(my_text)

    return plot_settings, text_settings


def plot_graph(vertices, edges, my_color='gray', number=False):
    """plot graph
    :param vertices : list of vertices locations (x,y)
    :param edges : list of connections (v1, v2)
    :param my_color : color for the graph (vertex and edge)
    :param number : set True to plot vertex number (ID) near each point"""

    # get edges index
    edges_idx = ext.edges_index(edges)

    settings = rpf.CustomizePlot()
    settings.set_marker('o')
    settings.set_l_color(my_color)

    # plot
    plot_custom_list(vertices, edges_idx, settings)

    settings.set_l_color('red')
    settings.set_marker('*')

    # for v in t0:
    #     point = vertices[v-1]
    #     # plot
    #     plot_custom_list(point, [], settings)

    if number:
        # plot index + 1 at v coordinates +- offset
        for coord in vertices:
            i = vertices.index(coord) + 1
            x = coord[0] + 0.5
            y = coord[1] + 0.75
            plt.text(x, y, str(i), fontsize=8, color=my_color)


def plot_explicit(vertices, edges, pi, settings, t, folder_name='coord'):
    """Plot graph with x, y coordinate"""

    plot_path_new(vertices, edges, pi, settings)

    # plt.show()

    fig_name = 'ex_' + str(t)
    f_name = save_plot(fig_name, folder_name)

    return f_name


def plot_implicit(vertices, edges, pi, settings, t, folder_name='coord'):

    plot_path_new(vertices, edges, pi, settings)

    # plt.show()

    fig_name = 'in_' + str(t)
    f_name = save_plot(fig_name, folder_name)

    return f_name


# def plot_implicit(vertices, edges, pi, settings, t):
#     plot_graph(vertices, edges)
#
#     s_main = 1
#     for s in pi.keys():
#
#         if s == s_main:
#             path = pi[s]
#         else:
#
#
#         path = pi[s]
#         s_edges = ext.path_conn(path)
#         plot_custom_list(vertices, s_edges, settings[s])
#
#     fig_name = 'im_' + t
#     save_plot(fig_name, 'coord')


def plot_path_new(vertices, edges, pi, settings):
    plot_graph(vertices, edges)

    for s in pi.keys():
        path = pi[s]
        s_edges = ext.path_conn(path)
        plot_custom_list(vertices, s_edges, settings[s])


def mount_coord_frame(ex_file, in_file, folder_path, t, settings, n_per_frame=1):

    im_1 = plt.imread(ex_file)
    im_2 = plt.imread(in_file)

    fig_1, ax_arr = plt.subplots(1, 2, figsize=(9, 5), dpi=400)

    for i in settings.text.keys():
        my_text = settings.text[i]
        my_pos = settings.text_pos[i]
        my_font = settings.fonts[i]
        fig_1.text(my_pos[0], my_pos[1], my_text, fontdict=my_font)

    ax_arr[0].imshow(im_1)
    ax_arr[1].imshow(im_2)

    # take out axis stuff
    for k in range(0, 2):
        ax_arr[k].set_xticklabels([])
        ax_arr[k].set_xticks([])
        ax_arr[k].set_yticklabels([])
        ax_arr[k].set_yticks([])
        ax_arr[k].axis('off')

    my_format = ".png"

    # save the frame
    n_start = 0
    for i in range(n_per_frame):
        frame_num = n_start + i + t * n_per_frame
        frame_string = str(frame_num)
        frame_string = frame_string.rjust(4, "0")

        fname = folder_path + "/" + "frame_" + frame_string + my_format

        # plt.figure(figsize=(4, 8), dpi=400)
        plt.savefig(fname, facecolor=None, edgecolor=None,
                    orientation='landscape', papertype=None,
                    transparent=True)
        plt.clf()

    plt.close('all')


def save_plot(fig_name: str, name_folder: str, parent_folder='data', my_ext='.png'):

    fig_path = ext.get_whole_path(name_folder, parent_folder) + '/' + fig_name + my_ext

    plt.savefig(fig_path, facecolor=None, edgecolor=None,
                orientation='landscape', transparent=True, bbox_inches='tight')

    # plt.show()
    plt.close()
    plt.clf()

    return fig_path


def define_text():
    my_words = rpf.empty_my_words(4)
    my_words[0]['text'] = 'Multi-Robot Search'
    my_words[1]['text'] = 'Team Coordination'
    my_words[2]['text'] = 'Explicit'
    my_words[3]['text'] = 'Implicit'
    my_words[0]['xy'] = (0.5, 0.88)
    my_words[1]['xy'] = (0.5, 0.83)
    my_words[2]['xy'] = (0.30, 0.05)
    my_words[3]['xy'] = (0.75, 0.05)

    return my_words


def plot_coordination(g, search_c: dict, search_d: dict, folder_name='coord'):

    # data folder path
    folder_path = ext.get_whole_path(folder_name)
    # get location of vertices and edges
    vertices, edges = set_location(g)

    plot_settings, text_settings = set_settings()

    my_words = define_text()

    f = len(search_c.keys())

    print("Starting plots")

    for t in range(f):
        ex_file = plot_explicit(vertices, edges, search_c[t], plot_settings, t)
        in_file = plot_explicit(vertices, edges, search_d[t], plot_settings, t)
        fig_paths = [ex_file, in_file]

        rpf.mount_coord_frame(fig_paths, t, my_words, 2)


def plot_coordination_simple(g, search_c: dict, search_d: dict, folder_name='coord_simple'):
    ext.get_whole_path(folder_name)
    # get location of vertices and edges
    vertices, edges = set_location(g)

    plot_settings, text_settings = set_settings()

    for k in search_c.keys():
        ex_file = plot_explicit(vertices, edges, search_c[k], plot_settings, k, folder_name)

    for k in search_d.keys():
        in_file = plot_implicit(vertices, edges, search_d[k], plot_settings, k, folder_name)
       # fig_paths = [ex_file, in_file]


if __name__ == "__main__":
    my_script()
    #  script_test()
    # create_graph()

