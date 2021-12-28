from setuptools import distutils
from inspect import getmembers, isfunction
import functools
import glob
import os
import pkgutil
import sys
import types

__all__ = [
    'get_python_library',
    'get_python_methods',
    'wraps',
]

if sys.version_info[0:2] >= (3, 4):  # Python v3.4+?
    wraps = functools.wraps  # built-in has __wrapped__ attribute
else:

    def wraps(wrapped,
              assigned=functools.WRAPPER_ASSIGNMENTS,
              updated=functools.WRAPPER_UPDATES):
        def wrapper(f):
            f = functools.wraps(wrapped, assigned, updated)(f)
            f.__wrapped__ = wrapped  # set attribute missing in earlier versions
            return f

        return wrapper


# unify functions
if sys.version_info[0] == 2:
    from itertools import izip as zip, imap as map, ifilter as filter

filter = filter
map = map
zip = zip
if sys.version_info[0] == 3:
    from functools import reduce

reduce = reduce
range = xrange if sys.version_info[0] == 2 else range

# unify itertools and functools
if sys.version_info[0] == 2:
    from itertools import ifilterfalse as filterfalse
    from itertools import izip_longest as zip_longest
    from functools32 import partial, wraps
else:
    from itertools import filterfalse
    from itertools import zip_longest
    from functools import partial, wraps


def get_python_library():

    # Get list of the loaded source modules on sys.path.
    modules = {
        module
        for _, module, package in list(pkgutil.iter_modules())
        if package is False
    }

    # Glob all the 'top_level.txt' files installed under site-packages.
    site_packages = glob.iglob(
        os.path.join(
            os.path.dirname(os.__file__) + '/site-packages', '*-info',
            'top_level.txt'))

    # Read the files for the import names and remove them from the modules
    # list.
    modules -= {open(txt).read().strip() for txt in site_packages}

    # Get the system packages.
    system_modules = set(sys.builtin_module_names)

    # Get the just the top-level packages from the python install.
    python_root = distutils.sysconfig.get_python_lib(standard_lib=True)
    _, top_level_libs, _ = list(os.walk(python_root))[0]

    return sorted(top_level_libs + list(modules | system_modules))


def get_python_methods(module):
    assert isinstance(module, types.ModuleType)
    return getmembers(module, isfunction)
