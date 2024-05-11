import os
import platform
import sys
import stat
import subprocess
import setuptools

if sys.version_info < (3,):
    raise Exception("Python 2 is not supported by MindSpeed.")

__description__ = 'MindSpeed for LLMs of Ascend'
__version__ = '0.0.1'
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


def req_file(filename):
    try:
        with open(filename) as f:
            content = f.readlines()
        res = [x.strip() for x in content]
    except FileNotFoundError:
        res = []
    return res


install_requires = req_file("requirements.txt")
cmd_class = {}
exts = []
flags = os.O_WRONLY | os.O_CREAT
modes = stat.S_IWUSR | stat.S_IRUSR | stat.S_IXUSR


def get_arch():
    arch_map = {
        'x86_64': 'x86_64',
        'arm64': 'aarch64',
        'aarch64': 'aarch64',
    }
    return arch_map.get(platform.machine(), "Unknown architecture")


def get_abi_version():
    try:
        import torch
        return '1' if torch.compiled_with_cxx11_abi() else '0'
    except ImportError as e:
        raise ImportError("PyTorch is not installed or there's an error importing it.") from e


def download_file(url, filename):
    try:
        import requests
        from tqdm import tqdm
        response = requests.head(url)
        total_size = int(response.headers.get('content-length', 0))

        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            with tqdm(total=total_size, unit='B', unit_scale=True, desc=filename) as bar:
                with os.fdopen(os.open(filename, flags, modes), 'wb') as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        bar.update(len(chunk))
                        f.write(chunk)
        return True
    except Exception as e:
        print(f"Error downloading file: {e}")
        return False


def atb_package():
    print("Enter Atb Package")
    arch = get_arch()
    if arch == "Unknown architecture":
        raise Exception("Unsupported architecture.")

    abi_version = get_abi_version()
    
    # atb蓝区下载链接，后续会依据版本号变动
    atb_url = f"https://pytorch-package.obs.cn-north-4.myhuaweicloud.com/cache/test/Ascend-mindie-atb_1.0.RC1_linux-{arch}_abi{abi_version}.run"
    atb_name = atb_url.split('/')[-1]
    print(f"Downloading {atb_name}...")
    if download_file(atb_url, atb_name):
        print(f'File downloaded to {atb_name}')
        os.chmod(atb_name, 0o755)
    else:
        raise Exception("Download Failed")

    print("Extracting ATB package...")
    extract_path = os.path.join(os.getcwd(), "mindspeed", "atb")
    os.makedirs(extract_path, exist_ok=True)
    absolute_atb_path = os.path.join(os.getcwd(), atb_name)
    subprocess.run(['./' + atb_name, '--noexec', '--extract=' + extract_path])
    
    init_file_path = os.path.join(extract_path, '__init__.py')
    with os.fdopen(os.open(init_file_path, flags, modes), 'w') as f:
        pass
        
    manifest_content = "recursive-include mindspeed/atb *"
    with os.fdopen(os.open('MANIFEST.in', flags, modes), 'w') as f:
        f.write(manifest_content)


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
    install_requires=install_requires,
    # Add in any packaged data.
    include_package_data=True,
    install_package_data=True,
    exclude_package_data={'': ['**/*.md']},
    package_data={'mindspeed': ['**/*.h', '**/*.cpp']},
    zip_safe=False,
    # PyPI package information.
    keywords=__keywords__,
    cmdclass={},
    ext_modules=exts
)

if os.getenv('ENABLE_ATB', '0') == '1':
    atb_package()