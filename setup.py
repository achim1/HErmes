import sys
import re
import os
import os.path
import stat
import shutil
import atexit

from glob import glob

from setuptools import setup
from setuptools.command.install import install



def _post_install():
    """
    Copy matplotlib style files and set the permissions right
    - at least for unix/sudo style systems

    Returns:
        None
    """
    print ("--------------------------------")
    print ("Running post install...")

    from matplotlib import get_configdir

    styles = sorted(glob("resources/*mplstyle"))

    as_root = False
    if os.getuid() == 0:
        try:
            uid = int(os.getenv("SUDO_UID"))
            gid = int(os.getenv("SUDO_GID"))
        except Exception as e:
            print ("Caught exception {}".format(e))
            print ("There is no SUDO_UID/SUDO_GID shellvariable...")
            as_root = True
            
            import pwd
            import subprocess as sub

            whois = sub.Popen(["who"], stdout=sub.PIPE).communicate()[0].split()[0]
            # python2/3
            if hasattr(whois, "decode"):
                whois = whois.decode()

            uid = int(pwd.getpwnam(whois).pw_uid)
            gid = int(pwd.getpwnam(whois).pw_gid)

    else:
        uid = int(os.getuid())
        gid = int(os.getgid())

    mplstylelib = get_configdir()
    mplstylelib = os.path.join(mplstylelib, "stylelib")
    if as_root:
        mplstylelib = mplstylelib.replace("/root", "/home/" + whois)
    if not os.path.exists(mplstylelib):
        print ("WARNING: Can not find stylelib dir {}".format(mplstylelib))
        print ("Creating {}".format(mplstylelib))
        os.mkdir(mplstylelib)
    for st in styles:

        print("INSTALLING {} to {}".format(st, mplstylelib))
        shutil.copy(st, mplstylelib)
        with open(os.path.join(mplstylelib, os.path.split(st)[1])) as fd:

            os.fchown(fd.fileno(), uid, gid)
            #os.fchmod(fd.fileno(), stat.S_IRWXU & stat.S_IRGRP & stat.S_IROTH)
            os.fchmod(fd.fileno(), 0o755)

class full_install(install):
    """
    Installation routine which executes post_install
    """

    def __init__(self, *args, **kwargs):
        #super(new_install, self).__init__(*args, **kwargs)
        install.__init__(self, *args, **kwargs)
        atexit.register(_post_install)


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
        
        return reqs

try:
    requirements = parse_requirements("requirements.txt")
except Exception as e:
    print ("Failed parsing requiremnts, installing dummy requirements...")
    requirements = ['numpy>=1.9.0',
                     'matplotlib>=1.5.0',
                     'pandas>=0.17.1',
                     'appdirs>=1.4.0',
                     'futures>=3.0.5',
                     'future>=0.16.0',
                     'pyprind>=2.9.6']

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
    cmdclass={'install': full_install},
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
      packages=['HErmes','HErmes.icecube_goodies',\
                'HErmes.plotting', 'HErmes.utils',\
                'HErmes.selection', 'HErmes.fitting',\
                'HErmes.analysis'],
      #scripts=[],
      package_data={'HErmes': [#'plotting/plotsconfig.yaml',\
                                #'plotting/pyevseldefault.mplstyle',\
                                #'plotting/pyevselpresent.mplstyle',\
                                'utils/PATTERNS.cfg',\
                                "icecube_goodies/geometry_ic86.h5"]}
      )
