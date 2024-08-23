#!/usr/bin/env python3

"""
``mindspeed `` provides a limited set of the functionality as ``git patch``.

Uasge
--------

1. Using mindspeed patch
++++++++++++++++++++++++++

To apply patches
::

    mindspeed -P
    or
    mindspeed --patch

2. Reverse mindspeed patch
++++++++++++++++++++++++++++

To reverse patches
::

    mindspeed -R
    or
    mindspeed --reverse


Add Patches
--------------

1. Use ``git diff xx.py > xx.patch`` to generate a single new patch.
2. Place it in the corresponding directory of MindSpeed.
3. It will be automatically found by 'def find_all_patch()'


Delete Patches
-----------------

Delete the patch in the corresponding directory of MindSpeed.


Rejects
----------

Option '--check' is used before patching.
If there is a reject when checking a patch, it will be passed, recorded and printed at the end.
You need to resolve them manualy.

"""
import subprocess
import os
from argparse import ArgumentParser

_RUN_PATH = os.path.dirname(__file__)


def find_all_patch(file_dir=None, target_suffix='.patch'):
    patch_files = []
    walk_generator = os.walk(file_dir)
    for root_path, dirs, files in walk_generator:
        if len(files) < 1:
            continue
        for file in files:
            file_name, suffix_name = os.path.splitext(file)
            if suffix_name == target_suffix:
                patch_files.append(os.path.join(root_path, file))
    return patch_files


def get_args_parser() -> ArgumentParser:
    '''Helper function parsing the commond line options'''

    parser = ArgumentParser(description="MindSpeed Patch Launcher")
    parser.add_argument(
        "-P",
        "--patch",
        action='store_true',
        help="Use mindspeed patch")
    parser.add_argument(
        "-R",
        "--reverse",
        action='store_true',
        help="Reverse mindspeed patch")
    return parser


def parse_args(args):
    parser = get_args_parser()
    return parser.parse_args(args)


def patch_from_args(args):
    rejects = []
    if args.patch:
        options = {'name': 'Applying', 'cmd': '-p1'}
    elif args.reverse:
        options = {'name': 'Reversing', 'cmd': '-R'}
    else:
        print('Hello MindSpeed')
        return

    patch_files = find_all_patch(os.path.dirname(_RUN_PATH))

    for patch in patch_files:
        print('{} patch {}...'.format(options['name'], patch))
        commond = 'git apply --check {} {}'.format(options['cmd'], patch)
        check = subprocess.run(commond.split(), capture_output=True, text=True)
        if check.stderr:
            rejects.append(patch)
            print('{} patch failed. Please check: {}'.format(options['name'], check.stderr))
        else:
            commond = 'git apply {} {}'.format(options['cmd'], patch)
            process = subprocess.run(commond.split(), capture_output=True, text=True)
            print(process.stdout)
    if rejects:
        print('Here are some rejects needed to resolve: {}'.format(rejects))


def main(args=None):
    print('MindSpeedRun Path is {}'.format(_RUN_PATH))
    args = parse_args(args)
    patch_from_args(args)


if __name__ == "__main__":
    main()