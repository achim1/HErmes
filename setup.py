import sys
import re
import os
import os.path

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

with open(os.path.join(os.path.dirname(__file__), 'README.md')) as readme:
    long_description = readme.read()


def parse_requirements(req_file):
    with open(req_file) as f:
        reqs = []
        for r in f.readlines():
            if not r.startswith("http"):
                reqs.append(r)
            elif ";" in r:
                continue # FIXME: find better solution
                #data = r.split(";")       
                #reqs.append(data[0]) 
        return reqs

no_parse_requirements = False

try:
    requirements = parse_requirements("requirements.txt")
except Exception as e:
    no_parse_requirements = True

if sys.version_info.major < 3:
    no_parse_requirements = True 
    
if no_parse_requirements:
    print ("Not parsing requiremnts.txt, installing requirements from list in setup.py...")
    requirements = ['numpy>=1.9.0',
                     'matplotlib>=1.5.0',
                     'pandas>=0.17.1',
                     'appdirs>=1.4.0',
                     'future>=0.16.0',
                     'pyprind>=2.9.6']


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

setup(name='HErmes',
      version=version,
      python_requires='>=3.6.0',
      description='Highly efficient, rapid multipurpose event selection',
      #long_description='Manages bookkeeping for different simulation datasets, developed for the use with IceCube data',
      long_description=long_description,
      author='Achim Stoessl',
      author_email="achim.stoessl@gmail.com",
      url='https://github.com/achim1/HErmes',
      #download_url="pip install HErmes",
      install_requires=requirements, 
      setup_requires=setup_requires,
      license="GPL",
      #cmdclass={'install': full_install},
      platforms=["Ubuntu 18.04", "Ubuntu 20.04"],
      classifiers=[
        "License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)",
        "Development Status :: 3 - Alpha",
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
