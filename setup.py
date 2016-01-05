from setuptools import setup

setup(name='pyevsel',
      version='0.1.0',
      description='Eventselection for HEP analysis',
      long_description='Manages bookkeeping for different simulation datasets, developed for the use with IceCube data',
      author='Achim Stoessl',
      author_email="achim.stoessl@gmail.com",
      url='https://github.com/achim1/pyevsel',
      download_url="pip install pyevsel",
      install_requires=['numpy>=1.9.0',
                        'matplotlib>=1.5.0'],
      license="GPL",
      platforms=["Ubuntu 12.04"],
      classifiers=[
        "License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)",
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 2.7",
        "Topic :: Scientific/Engineering :: Physics"
              ],
      keywords=["event selection", "physics",\
                "hep", "particle physics"\
                "astrophysics", "icecube"],
      packages=['pyevsel','pyevsel.icecube_goodies',\
                'pyevsel.plotting','pyevsel.utils',\
                'pyevsel.variables']
      #scripts=['bin/muonic','bin/which_tty_daq'],
      #package_data={'muonic': ['daq/simdaq.txt'],'':['*.txt','*.rst']},
      #package_data={'' : ['docs/*','README'], 'muonic': ['daq/simdaq.txt','daq/which_tty_daq']}, 
      )
