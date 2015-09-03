import os
import sys
import timeit
import settings


def add_to_path(p):
    if p not in sys.path:
        sys.path.insert(0, p)


def add_caffe_to_path():
    caffe_python_root = os.path.join(settings.CAFFE_ROOT, 'python')
    add_to_path(caffe_python_root)


class Timer:
    def __init__(self, message='Execution'):
        self.message = message

    def __enter__(self):
        self.start = timeit.default_timer()
        return self

    def __exit__(self, *args):
        self.end = timeit.default_timer()
        self.interval = self.end - self.start

        print '{} took {:.2f} seconds'.format(self.message, self.interval)


def ensuredir(dirpath):
    if not os.path.exists(dirpath):
        os.makedirs(dirpath)


def plot_2D_arrays(arrs, title='', xlabel='', xinterval=None, ylabel='', yinterval=None, line_names=[], simplified=False):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import seaborn as sns

    plt.clf()
    sns.set_style('darkgrid')

    for arr in arrs:
        if arr.ndim != 2 or arr.shape[1] != 2:
            raise ValueError('The array should be 2D and the second dimension should be 2!')

        plt.plot(arr[:, 0], arr[:, 1])

    # If simplified, we don't write text anywhere
    if not simplified:
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        if line_names:
            plt.legend(line_names, loc=6, bbox_to_anchor=(1, 0.5))

    if xinterval:
        plt.xlim(xinterval)
    if yinterval:
        plt.ylim(yinterval)

    plt.tight_layout()


def plot_and_save_2D_arrays(filename, arrs, xlabel='', xinterval=None, ylabel='', yinterval=None, line_names=[], simplified=False):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    name, ext = os.path.splitext(os.path.basename(filename))
    plot_2D_arrays(arrs, name, xlabel, xinterval, ylabel, yinterval, line_names, simplified)
    plt.savefig(filename)
    plt.clf()


def plot_and_save_2D_array(filename, arr, xlabel='', xinterval=None, ylabel='', yinterval=None, simplified=False):
    plot_and_save_2D_arrays(filename, [arr], xlabel, xinterval, ylabel, yinterval, line_names=[], simplified=simplified)
