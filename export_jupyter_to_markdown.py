from argparse import ArgumentParser
import glob
import os

import argparse

parser = ArgumentParser()
parser.add_argument('--override', action='store_true', default=False,
                    help='Indicated whether override existing markdown files or not.')
args = parser.parse_args()

all_jupyter_files = glob.glob("tutorials/*/*.ipynb")
all_markdown_files = [i[:-6] + ".md" for i in all_jupyter_files]

for ipynb, md in zip(all_jupyter_files, all_markdown_files):
    if not args.override and os.path.exists(md):
        continue
    cmd = "jupyter nbconvert --to markdown {}".format(ipynb)
    stat = os.system(cmd)
    if stat == 0:
        print("Convert '{}' to markdown format success.".format(ipynb))
    else:
        print("Convert '{}' to markdown format fail.".format(ipynb))
