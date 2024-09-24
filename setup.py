import os
import platform
import sys
import stat
import subprocess
import setuptools

if sys.version_info < (3,):
    raise Exception("Python 2 is not supported by MindSpeed.")

__description__ = 'MindSpeed for LLMs of Ascend'
__version__ = '0.7.0'
__author__ = 'Ascend'
__long_description__ = 'MindSpeed for LLMs of Ascend'
__url__ = 'https://gitee.com/ascend/MindSpeed'
__download_url__ = 'https://gitee.com/ascend/MindSpeed/release'
__keywords__ = 'Ascend, langauge, deep learning, NLP'
__license__ = 'See https://gitee.com/ascend/MindSpeed'
__package_name__ = 'mindspeed'
__contact_names__ = 'Ascend'

try:
    with open("README.md", "r") as fh:
        long_description = fh.read()
except FileNotFoundError:
    long_description = ''


###############################################################################
#                             Dependency Loading                              #
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #

cmd_class = {}
exts = []


def package_files(directory):
    paths = []
    for path, directories, filenames in os.walk(directory):
        for filename in filenames:
            paths.append(os.path.join(path, filename))
    return paths


src_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'mindspeed')

if os.getenv('CI_BUILD', '0') != '1':
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt'])

setuptools.setup(
    name=__package_name__,
    # Versions should comply with PEP440.  For a discussion on single-sourcing
    # the version across setup.py and the project code, see
    # https://packaging.python.org/en/latest/single_source_version.html
    version=__version__,
    description=__description__,
    long_description=long_description,
    long_description_content_type="text/markdown",
    # The project's main homepage.
    url=__url__,
    author=__contact_names__,
    maintainer=__contact_names__,
    # The licence under which the project is released
    license=__license__,
    classifiers=[
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'Intended Audience :: Information Technology',
        # Indicate what your project relates to
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Software Development :: Libraries :: Python Modules',
        # Supported python versions
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        # Additional Setting
        'Environment :: Console',
        'Natural Language :: English',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
    packages=setuptools.find_packages(),
    # Add in any packaged data.
    include_package_data=True,
    install_package_data=True,
    exclude_package_data={'': ['**/*.md']},
    package_data={'': package_files(src_path)},
    bug_data={'mindspeed': ['**/*.h', '**/*.cpp', '*/*.sh', '**/*.patch']},
    zip_safe=False,
    # PyPI package information.
    keywords=__keywords__,
    cmdclass={},
    entry_points={
        "console_scripts": [
            "mindspeed = mindspeed.run.run:main",
        ]
    },
    ext_modules=exts
)
