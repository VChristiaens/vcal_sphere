#!/usr/bin/env python

import os
import re
from setuptools import setup
try:  # pip >= 10
    from pip._internal.req import parse_requirements
except ImportError:  # pip <= 9.0.3
    from pip.req import parse_requirements
from setuptools.command.install import install
from setuptools.command.develop import develop


# Hackishly override of the install method
class InstallReqs(install):
    def run(self):
        print(" ****************************** ")
        print(" *** Installing VCAL-SPHERE *** ")
        print(" ****************************** ")
        os.system('pip install -r requirements.txt')
        install.run(self)


class InstallDevReqs(develop):
    def run(self):
        print(" ****************************** ")
        print(" *** Installing VCAL-SPHERE *** ")
        print(" ****************************** ")
        os.system('pip install -r requirements-dev.txt')
        develop.run(self)


def resource(*args):
    return os.path.join(os.path.abspath(os.path.join(__file__, os.pardir)),
                        *args)


# parse_requirements() returns generator of pip.req.InstallRequirement objects
reqs = parse_requirements(resource('requirements.txt'), session=False)
try:
    reqs = [str(ir.req) for ir in reqs]
except:
    reqs = [str(ir.requirement) for ir in reqs]
reqs_dev = parse_requirements(resource('requirements-dev.txt'), session=False)
try:
    reqs_dev = [str(ir.req) for ir in reqs_dev]
except:
    reqs_dev = [str(ir.requirement) for ir in reqs_dev]    

with open(resource('README.rst')) as readme_file:
    README = readme_file.read()

with open(resource('vcal', '__init__.py')) as version_file:
    version_file = version_file.read()
    VERSION = re.search(r"""^__version__ = ['"]([^'"]*)['"]""",
                        version_file, re.M)
    VERSION = VERSION.group(1)

# MODIFY EXAMPLE json file to include path to static data:
curr_dir = os.getcwd()
curr_dir+="/"
old_names = ["Examples/VCAL_params_calib_IFS.json",
             "Examples/VCAL_params_calib_IRDIS.json"]
new_names = ["Examples/VCAL_params_calib_IFS_new.json",
            "Examples/VCAL_params_calib_IRDIS_new.json"]
for nn in range(len(old_names)):
    old_name = old_names[nn]
    new_name = new_names[nn]
    with open(curr_dir+old_name, "r+") as fold:
        with open(curr_dir+new_name, "w") as fnew:
            all_lines = fold.readlines()
            for ll in range(len(all_lines)):
                if ll !=2:
                    fnew.write(all_lines[ll])
                else:
                    line_tmp = '    "inpath_filt_table": "{}{}",\n'
                    filt_name = 'sph_ird_filt_table.fits'
                    new_line = line_tmp.format(curr_dir+"DataStatic/", filt_name)
                    fnew.write(new_line)
    os.system("rm {}{}".format(curr_dir, old_name))
    os.system("mv {}{} {}{}".format(curr_dir, new_name, curr_dir, old_name))

PACKAGES = ['vcal',
            'vcal.calib',
            'vcal.postproc',
            'vcal.preproc',
            'vcal.utils']

setup(
    name='vcal',
    version=VERSION,
    description='Package for VIP-based SPHERE image Calibration and processing',
    long_description=README,
    license='MIT',
    author='Valentin Christiaens',
    author_email='valentinchrists@hotmail.com',
    url='https://github.com/vortex-exoplanet/VIP',
    cmdclass={'install': InstallReqs,
              'develop': InstallDevReqs},
    packages=PACKAGES,
    install_requires=reqs,
    extras_require={"dev": reqs_dev},
    zip_safe=False,
    classifiers=['Intended Audience :: Science/Research',
                 'License :: OSI Approved :: MIT License',
                 'Operating System :: MacOS :: MacOS X',
                 'Operating System :: POSIX :: Linux',
                 'Natural Language :: English',
                 'Programming Language :: Python :: 3.6',
                 'Programming Language :: Python :: 3.7',
                 'Topic :: Scientific/Engineering :: Astronomy'
                 ]
)
