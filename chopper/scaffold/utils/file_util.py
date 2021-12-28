import os
from datetime import datetime

__all__ = [
    'read_src',
    'dump_to_file',
]


def get_time_stamp():
    _t = datetime.now()
    return _t.strftime("%Y%m%d%H%M")


def read_src(filename):
    # type: (basestring) -> basestring
    with open(filename, 'r') as fp:
        return fp.read()


def dump_to_file(filename, text, prefix="anonymous"):
    # type: (basestring, basestring) -> None
    os.system('mkdir -p dump_ast')
    dirpath = os.path.join(
        os.getcwd(), "./dump_ast/" + prefix + '_' + get_time_stamp() + '/')
    filepath = dirpath + str(filename)
    os.system('mkdir -p ' + dirpath)
    with open(os.path.join("./dump_ast", filepath), 'w') as fp:
        fp.write(text)
    return


if __name__ == '__main__':
    dump_to_file('test.ast', "\nprint('hello world')\n")
