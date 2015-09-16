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
