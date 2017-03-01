from setuptools import setup

from pyevsel import __version__

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

requirements.append("tables>=3.3.0") # problem with travis CI, removed from requirments.txt

setup(name='pyevsel',
      version=__version__,
      description='Eventselection for HEP analysis',
      long_description='Manages bookkeeping for different simulation datasets, developed for the use with IceCube data',
      author='Achim Stoessl',
      author_email="achim.stoessl@gmail.com",
      url='https://github.com/achim1/pyevsel',
      #download_url="pip install pyevsel",
      install_requires=requirements, 
      setup_requires=['pytest-runner'],
      license="GPL",
      platforms=["Ubuntu 14.04","Ubuntu 16.04"],
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
      tests_require=['pytest'],
      packages=['pyevsel','pyevsel.icecube_goodies',\
                'pyevsel.plotting','pyevsel.utils',\
                'pyevsel.variables'],
      #scripts=[],
      package_data={'pyevsel': ['plotting/plotsconfig.yaml','plotting/pyevseldefault.mplstyle','plotting/pyevselpresent.mplstyle','utils/PATTERNS.cfg']}
      )
