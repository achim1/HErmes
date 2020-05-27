"""
HErmes setup.py file. To install, run python setup.py install. Requirese python >=3.6

"""

import sys
import re
import os
import os.path
import pathlib
import pkg_resources as pk
from setuptools import setup


def is_tool(name):
    """Check whether `name` is on PATH."""

    from distutils.spawn import find_executable
    return find_executable(name) is not None

# get_version and conditional adding of pytest-runner
# are taken from 
# https://github.com/mark-adams/pyjwt/blob/b8cc504ee09b4f6b2ba83a3db95206b305fe136c/setup.py

def get_version(package):
    """
    Return package version as listed in `__version__` in `init.py`.
    """
    with open(os.path.join(package, '__init__.py'), 'rb') as init_py:
        src = init_py.read().decode('utf-8')
        return re.search("__version__ = ['\"]([^'\"]+)['\"]", src).group(1)

version = get_version('HErmes')

long_description = "Aggregate data from hdf or root files and conveniently apply filter criterias.\
Explore datasets with ease with a focus on interactivity and rapidity."

#parse the requirements.txt file
# FIXME: this might not be the best way
install_requires = []
with pathlib.Path('requirements.txt').open() as requirements_txt:
    for line in requirements_txt.readlines():
        if line.startswith('#'):
            continue
        try:
            req = str([j for j in pk.parse_requirements(line)][0])
        except Exception as e:
            print (f'WARNING: {e} : Can not parse requirement {line}')
            continue
        install_requires.append(req)

h5ls_available = is_tool('h5ls')
if not h5ls_available:
    print ("ERROR: h5ls is not installed. This will cause MASSIVE problems in case you intend to work with hdf files.")

#requirements.append("tables>=3.3.0") # problem with travis CI, removed from requirments.txt

tests_require = [
    'pytest>=3.0.5',
    'pytest-cov',
    'pytest-runner',
]

needs_pytest = set(('pytest', 'test', 'ptr')).intersection(sys.argv)
setup_requires = ['pytest-runner'] if needs_pytest else []
#setup_requires += ["matplotlib>=1.5.0"]

setup(name='HErmes-py',
      version=version,
      python_requires='>=3.6.0',
      description='Highly efficient, rapid multipurpose event selection',
      #long_description='Manages bookkeeping for different simulation datasets, developed for the use with IceCube data',
      long_description=long_description,
      author='Achim Stoessl',
      author_email="achim.stoessl@gmail.com",
      url='https://github.com/achim1/HErmes',
      #download_url="pip install HErmes",
      install_requires=install_requires, 
      setup_requires=setup_requires,
      license="GPL",
      #cmdclass={'install': full_install},
      platforms=["Ubuntu 18.04", "Ubuntu 20.04"],
      classifiers=[
        "License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)",
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.8",
        "Topic :: Scientific/Engineering :: Physics"
              ],
      keywords=["event selection", "physics",\
                "hep", "particle physics"\
                "astrophysics", "icecube"],
      tests_require=tests_require,
      packages=['HErmes','HErmes.icecube_goodies',\
                'HErmes.visual', 'HErmes.utils',\
                'HErmes.selection', 'HErmes.fitting',\
                'HErmes.analysis'],
      #scripts=[],
      package_data={'HErmes': [ 'utils/PATTERNS.cfg',\
                                "icecube_goodies/geometry_ic86.h5"]}
      )
