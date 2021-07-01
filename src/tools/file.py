import os

from matplotlib import pyplot as plt

from src.tools.constants import REPORT_IMAGE_DIR


def make_dirs(file_name):
    dir_name = os.path.dirname(file_name)
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)


def get_plot_path(image_name):
    return os.path.join(REPORT_IMAGE_DIR, f'{image_name}.svg')


def store_plt(image_name):
    plt.savefig(get_plot_path(image_name), format="svg")
