import sys
from setuptools import setup

# Written according to the docs at
# https://packaging.python.org/en/latest/distributing.html

if sys.version_info[0] < 3:
    sys.exit('This script requires python 3.0 or higher to run.')

setup(
    name='olcaoPy',
    description=('A high level package to manipulate atomic structures '
                 'and to run/customize the OLCAO package'),
    version='0.1.0',
    url='https://github.com/NDari/olcaoPy',
    author='Naseer Dari, James Currie, Paul Rulis',
    author_email='naseerdari01@gmail.com',
    license='MIT',
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'Topic :: Education',
        'Topic :: Scientific/Engineering'
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.2',
        'Programming Language :: Python :: 3.3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6'
    ],
    packages=['control', 'symmetryFunctions', 'bondAnalysis', 'olcao', 'constants'],
    install_requires=['numpy']
)
