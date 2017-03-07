import sys
import re
import os.path

from setuptools import setup

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

version = get_version('pyevsel')

with open(os.path.join(os.path.dirname(__file__), 'README.md')) as readme:
    long_description = readme.read()


def parse_requirements(req_file):
    with open(req_file) as f:
        reqs = []
        for r in f.readlines():
            if not r.startswith("http"):
                reqs.append(r)
        
        return reqs

try:
    requirements = parse_requirements("requirements.txt")
except Exception as e:
    print ("Failed parsing requiremnts, installing dummy requirements...")
    requirements = ['numpy>=1.9.0',
                     'matplotlib>=1.5.0',
                     'pandas>=0.17.1',
                     'ruamel.yaml>=0.13.14',
                     'appdirs>=1.4.0',
                     'futures>=3.0.5',
                     'future>=0.16.0',
                     'pyprind>=2.9.6']

#requirements.append("tables>=3.3.0") # problem with travis CI, removed from requirments.txt

tests_require = [
    'pytest>=2.7.3',
    'pytest-cov',
    'pytest-runner',
]

needs_pytest = set(('pytest', 'test', 'ptr')).intersection(sys.argv)
setup_requires = ['pytest-runner'] if needs_pytest else []
setup_requires += ["matplotlib>=1.5"]

setup(name='pyevsel',
      version=version,
      description='Eventselection for HEP analysis',
      #long_description='Manages bookkeeping for different simulation datasets, developed for the use with IceCube data',
      long_description=long_description,
      author='Achim Stoessl',
      author_email="achim.stoessl@gmail.com",
      url='https://github.com/achim1/pyevsel',
      #download_url="pip install pyevsel",
      install_requires=requirements, 
      setup_requires=setup_requires,
      license="GPL",
      platforms=["Ubuntu 14.04","Ubuntu 16.04", "Ubuntu 16.10", "SL6.1"],
      classifiers=[
        "License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)",
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 2.7",
        "Programming Language :: Python :: 3.5",
        "Topic :: Scientific/Engineering :: Physics"
              ],
      keywords=["event selection", "physics",\
                "hep", "particle physics"\
                "astrophysics", "icecube"],
      tests_require=tests_require,
      packages=['pyevsel','pyevsel.icecube_goodies',\
                'pyevsel.plotting','pyevsel.utils',\
                'pyevsel.variables', 'pyevsel.fitting'],
      #scripts=[],
      package_data={'pyevsel': ['plotting/plotsconfig.yaml','plotting/pyevseldefault.mplstyle','plotting/pyevselpresent.mplstyle','utils/PATTERNS.cfg']}
      )
