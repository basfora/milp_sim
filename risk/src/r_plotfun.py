# -----------------------------------------------------------------------------------------
# Plot functions for MILP MESPP RISK-AWARE
# -----------------------------------------------------------------------------------------
import os
from matplotlib import pyplot as plt
# TODO IMPORT FROM THE ORIGINAL MILP_MESPP REPO


def compose_and_clean(path_to_folder: str):
    compose_video(path_to_folder)
    delete_frames(path_to_folder)


def compose_video(path_to_folder: str):
    """Use plots as frames and make a short video"""
    # print(path_now)
    print("Composing video")
    command_to_run = "ffmpeg -r 20 -f image2 -i " + path_to_folder + \
                     "/frame_%04d.png -vcodec libx264 -crf 18 -pix_fmt yuv420p " \
                     + path_to_folder + "/a_no_sync.mp4 -y"
    os.system(command_to_run)
    print("Video is done")


def delete_frames(path_to_folder: str, key_name='frame'):
    """Delete frames used to make the video to save space"""

    for filename in os.listdir(path_to_folder):
        if filename.startswith(key_name) and filename.endswith('.png'):
            my_file = path_to_folder + "/" + filename
            os.remove(my_file)

    print("Frames were deleted")
    return


def define_fonts():
    """Define dictionary with fonts for plotting"""

    n_fts = 4

    my_font = create_dict(list(range(0, n_fts)), None)
    my_sizes = create_dict(list(range(0, n_fts)), 10)

    # font and sizes
    my_sizes[0] = 13
    my_sizes[1] = 12
    my_sizes[2] = 11
    my_sizes[3] = 9

    # title - red and bold
    my_font[0] = {'family': 'serif', 'color': 'darkred', 'weight': 'bold', 'size': my_sizes[0],
                  'horizontalalignment': 'center'}

    # subtitle - dark blue and bold
    my_font[1] = {'family': 'serif', 'color': 'darkblue', 'weight': 'bold', 'size': my_sizes[1],
                  'horizontalalignment': 'center'}
    # regular text - gray
    my_font[2] = {'family': 'serif', 'color': 'darkgray', 'weight': 'normal', 'size': my_sizes[3],
                  'horizontalalignment': 'center'}

    # regular text - black
    my_font[3] = {'family': 'serif', 'color': 'black', 'weight': 'normal', 'size': my_sizes[3],
                  'horizontalalignment': 'center'}

    return my_font


def define_fonts2():
    my_font = define_fonts()

    my_font[2] = {'family': 'serif', 'color': 'black', 'weight': 'bold', 'size': 11,
                  'horizontalalignment': 'center'}\

    my_font[3] = my_font[2]

    return my_font

def place_image(im, ax_arr, my_subs: list):
    """ place image in subplot and take out ticks and stuff"""

    if len(my_subs) > 1:
        # plt.tight_layout()
        new_left = 0.1
        new_right = 0.9
        new_bottom = 0.1
        new_top = 0.9
        plt.subplots_adjust(wspace=0.1, hspace=0, left=new_left, right=new_right, bottom=new_bottom, top=new_top)

        for k in my_subs:
            ax_arr[k].imshow(im[k])
            ax_arr[k].set_xticklabels([])
            ax_arr[k].set_xticks([])
            ax_arr[k].set_yticklabels([])
            ax_arr[k].set_yticks([])
            ax_arr[k].axis('off')
    else:

        ax_arr.imshow(im[0])

        ax_arr.set_xticklabels([])
        ax_arr.set_xticks([])
        ax_arr.set_yticklabels([])
        ax_arr.set_yticks([])
        ax_arr.axis('off')

    return


def mount_frame(path_and_fig, t: int, my_words: dict, n_sub=1, video=False):
    """Mount frame for video
    :path_and_fig: path+name(s) of figure(s)
    :t: time step
    :my_words: dict with 3 keys in order my_title, unit of time, subtitle
    :n_sub: number of subplots"""

    # ----------------
    my_font = define_fonts()
    # -----------------

    # how many subplots
    my_subs = list(range(0, n_sub))

    # create figure with subplots
    fig_1, ax_arr = plt.subplots(1, n_sub, figsize=(9, 5), dpi=600)

    # retrieve graph plots as images
    im = {}
    if n_sub == 1:
        if isinstance(path_and_fig, str):
            im[0] = plt.imread(path_and_fig)
        else:
            im[0] = plt.imread(path_and_fig[2])

    else:
        for i in my_subs:
            im[i] = plt.imread(path_and_fig[i])

    # -------------------
    # plot text
    # insert time step
    my_words[0]['text'] = my_words[0]['text'] + str(t)

    # title and subtitle
    for line in range(0, 2):
        my_text = my_words[line]['text']
        x, y = my_words[line]['xy']

        fig_1.text(x, y, my_text, fontdict=my_font[line])

    if n_sub == 3:
        for col in range(1, 5):
            my_text = my_words[col]['text']
            x, y = my_words[col]['xy']

            # same for all cols
            idx = 1
            fig_1.text(x, y, my_text, fontdict=my_font[idx])

        for col in range(5, 11):
            my_text = my_words[col]['text']
            x, y = my_words[col]['xy']

            # same for all sub cols
            idx = 2
            fig_1.text(x, y, my_text, fontdict=my_font[idx])

    # labels
    my_hazard_labels(fig_1)
    # cax = plt.axes([0.85, 0.1, 0.075, 0.8])
    # plt.colorbar(cax=cax)
    # # -------------------

    # take out axis stuff
    place_image(im, ax_arr, my_subs)
    # -------------------

    # save in folder
    frame_idx = t
    save_frame(path_and_fig, frame_idx, video)


def save_frame(path_and_fig, frame_idx, video=False, key='hazard'):
    path_to_folder = os.path.dirname(path_and_fig[0])

    if video:
        save_copied_frames(path_to_folder, frame_idx)
    else:
        save_one_frame(path_to_folder, frame_idx, key)

    return


def save_one_frame(path_to_folder, frame_idx, key='hazard'):
    my_format = ".png"

    frame_num = frame_idx
    frame_string = str(frame_num)
    frame_string = frame_string.rjust(4, "0")

    fname = path_to_folder + "/" + key + "_" + frame_string + my_format

    # plt.figure(figsize=(4, 8), dpi=400)
    plt.savefig(fname, facecolor=None, edgecolor=None,
                orientation='landscape', papertype=None,
                transparent=True)


def save_copied_frames(path_to_folder: str, frame_idx: int, n_frames_per=60):
    """Multiply frames for video"""

    my_format = ".png"

    # save the frame
    # change n_start to 140 for complete video 140  # n_frame_per * 3
    n_start = 0
    for i in range(n_frames_per):
        frame_num = n_start + i + frame_idx * n_frames_per
        frame_string = str(frame_num)
        frame_string = frame_string.rjust(4, "0")

        fname = path_to_folder + "/" + "frame_" + frame_string + my_format

        # plt.figure(figsize=(4, 8), dpi=400)
        plt.savefig(fname, facecolor=None, edgecolor=None, orientation='landscape', transparent=True)


def my_hazard_labels(fig_1, xy=None, f_size=12):

    levels = [1, 2, 3, 4, 5]
    # level_label = ['Low', 'Moderate', 'High', 'Very High', 'Extreme']
    # level_color = ['green', 'blue', 'yellow', 'orange', 'red']
    level_color = color_levels(True)
    level_label = label_levels()

    my_font = {}

    if xy is None:
        x, y = 0.31, 0.1
    else:
        x, y = xy[0], xy[1]

    for i in levels:

        my_font[i] = {'family': 'sans-serif', 'color': level_color[i], 'weight': 'heavy', 'size': f_size,
                'horizontalalignment': 'left'}

        fig_1.text(x, y, level_label[i], fontdict=my_font[i])

        y = y + 0.08

    return fig_1


def empty_my_words(n: int):
    my_words = create_dict(list(range(0, n)), None)

    for i in range(0, n):
        my_words[i] = {'text': '', 'xy': (0.0, 0.0)}

    return my_words


def color_levels(normalized=False):
    """Change colors here"""

    level = create_dict([1, 2, 3, 4, 5], None)

    # (R, G, B)
    # yellow = (1, 1, 0)
    # orange =

    # green
    level[1] = (60, 180, 60)
    # yellow-green
    level[2] = (200, 200, 30)
    # yellow
    level[3] = (240, 215, 40)
    # orange
    level[4] = (250, 120, 50)
    # red
    level[5] = (255, 30, 30)

    if normalized is True:
        for k in level.keys():
            r = level[k][0]
            g = level[k][1]
            b = level[k][2]
            level[k] = (r/255, g/255, b/255)
    return level


def label_levels():
    labels = create_dict(list(range(1, 6)), '')

    labels[1] = 'Low'
    labels[2] = 'Moderate'
    labels[3] = 'High'
    labels[4] = 'Very High'
    labels[5] = 'Extreme'

    return labels


def match_level_color(my_level: int):
    """Define color for levels"""

    colors = color_levels(True)

    my_color = set_vertex_color(colors[my_level])

    return my_color


def set_vertex_color(my_color: tuple):

    alpha = 1
    red, green, blue = my_color
    return red, green, blue, alpha


def create_dict(my_keys, default_value):

    if isinstance(default_value, dict):
        print('Starting default value as {} will cause everything to have the same value')
        default_value = None

    my_dict = {}
    for k in my_keys:
        my_dict[k] = default_value
    return my_dict


# ------------------------------------------------------------
def mount_coord_frame(path_and_fig, t: int, my_words: dict, n_sub=1, video=False):
    """Mount frame for video
    :path_and_fig: path+name(s) of figure(s)
    :t: time step
    :my_words: dict with 3 keys in order my_title, subtitle, left title, right title
    :n_sub: number of subplots"""

    # ----------------
    my_font = define_fonts2()
    # -----------------

    # how many subplots
    my_subs = list(range(0, n_sub))

    # create figure with subplots
    fig_1, ax_arr = plt.subplots(1, n_sub, figsize=(9, 5), dpi=600)

    # retrieve graph plots as images
    im = {}
    if n_sub == 1:
        if isinstance(path_and_fig, str):
            im[0] = plt.imread(path_and_fig)
        else:
            im[0] = plt.imread(path_and_fig[2])

    else:
        for i in my_subs:
            im[i] = plt.imread(path_and_fig[i])

    # -------------------
    # plot text
    # insert time step

    # title and subtitle
    for line in my_words.keys():
        my_text = my_words[line]['text']
        x, y = my_words[line]['xy']

        fig_1.text(x, y, my_text, fontdict=my_font[line])

    if n_sub == 3:
        for col in range(1, 5):
            my_text = my_words[col]['text']
            x, y = my_words[col]['xy']

            # same for all cols
            idx = 1
            fig_1.text(x, y, my_text, fontdict=my_font[idx])

        for col in range(5, 11):
            my_text = my_words[col]['text']
            x, y = my_words[col]['xy']

            # same for all sub cols
            idx = 2
            fig_1.text(x, y, my_text, fontdict=my_font[idx])

    # labels
    # my_hazard_labels(fig_1)
    # cax = plt.axes([0.85, 0.1, 0.075, 0.8])
    # plt.colorbar(cax=cax)
    # # -------------------

    # take out axis stuff
    place_image(im, ax_arr, my_subs)
    # -------------------

    # save in folder
    frame_idx = t
    save_frame(path_and_fig, frame_idx, video, 'frame')


class CustomizePlot:

    def __init__(self):

        # line specs
        self.line_style = 'solid'
        self.line_color = 'k'
        self.line_width = 2

        # marker specs
        self.marker = None
        self.marker_color = self.line_color
        self.marker_size = 5

        # title
        self.title = None
        self.subtitle = None
        self.fonts = self.set_fonts()

        # other text
        self.text = None
        self.text_pos = self.set_pos()

        # axis
        self.xlabel = None
        self.ylabel = None

        # legend
        self.legend = None

        # orientation
        self.orientation = 'landscape'

    @staticmethod
    def set_fonts():

        size_title = 13
        size_subtitle = 12
        size_text = 10

        my_fonts = dict()

        my_fonts[0] = {'family': 'serif', 'color': 'darkblue', 'weight': 'normal', 'size': size_subtitle,
                       'horizontalalignment': 'center'}
        my_fonts[1] = {'family': 'serif', 'color': 'darkred', 'weight': 'normal', 'size': size_title,
                       'horizontalalignment': 'center'}
        my_fonts[2] = {'family': 'serif', 'color': 'darkgray', 'weight': 'normal', 'size': size_text,
                       'horizontalalignment': 'center'}
        my_fonts[3] = {'family': 'serif', 'color': 'black', 'weight': 'normal', 'size': size_title,
                       'horizontalalignment': 'center'}
        my_fonts[4] = {'family': 'serif', 'color': 'black', 'weight': 'normal', 'size': size_title,
                       'horizontalalignment': 'center'}

        return my_fonts

    def set_pos(self, list_pos=None):

        positions = dict()

        if list_pos is None:
            positions[0] = (0.5, 0.93)
            positions[1] = (0.5, 0.88)
            positions[2] = (0.5, 0.83)
            positions[3] = (0.30, 0.05)
            positions[4] = (0.75, 0.05)
            return positions

        else:
            i = 0
            for el in list_pos:
                positions[i] = el
                i += 1

        self.text_pos = positions

    def set_l_style(self, my_style: str):
        self.line_style = my_style

    def set_l_color(self, my_color: str):
        self.line_color = my_color
        self.marker_color = my_color

    def set_l_width(self, my_w: int):
        self.line_width = my_w

    def set_marker(self, my_marker: str):
        self.marker = my_marker

    def set_marker_size(self, my_size: int):
        self.marker_size = my_size

    def set_marker_color(self, my_color: str):
        self.marker_color = my_color

    def set_text(self, my_text: dict):
        self.text = my_text



