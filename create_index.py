import os
import re
import argparse
import sys
from jinja2 import Environment, FileSystemLoader, select_autoescape

# url prefix for the links
prefix = "https://nbviewer.org/urls/qutip.org/qutip-tutorials/"

# tutorial directories
tutorial_directories = ['time-evolution', 'lectures']

# Create an argument to the script that contains the tutorials
parser = argparse.ArgumentParser(
    description='Generate Index file for QuTiP Tutorials')
parser.add_argument('directory', type=str,
                    help='Directory containing the tutorial folders '
                         '(e.g. tutorials-v4)')
args = parser.parse_args()

# read the directory from the arguments
directory = args.directory
if not directory[-1] == '/':
    directory = directory + '/'
# Check if directory exists
if not os.path.exists(directory):
    print(
        'Given directory does not exists, please specify an existing directory')
    sys.exit(1)


def atoi(text):
    return int(text) if text.isdigit() else text


def natural_keys(text):
    return [atoi(c) for c in re.split('(\d+)', text)]


class notebook:
    def __init__(self, path, title):
        self.path = path
        self.title = title
        # set url and update from markdown to ipynb
        self.url = prefix + path.replace(".md", ".ipynb")


def get_title(filename):
    """ Reads the title from a markdown notebook """
    with open(filename, 'r') as f:
        # get first row that starts with "# "
        for line in f.readlines():
            # trim leading/trailing whitespaces
            line = line.lstrip().rstrip()
            # check if line is the title
            if line[0:2] == '# ':
                # return title
                return line[2:]


def sort_files_titles(files, titles):
    """ Sorts the files and titles either by filenames or titles """
    # identify numbered files and sort them
    nfiles = [s for s in files if s.split('/')[-1][0].isdigit()]
    nfiles = sorted(nfiles, key=natural_keys)
    ntitles = [titles[files.index(s)] for s in nfiles]
    # sort the files without numbering by the alphabetic order of the titles
    atitles = [titles[files.index(s)] for s in files if s not in nfiles]
    atitles = sorted(atitles, key=natural_keys)
    afiles = [files[titles.index(s)] for s in atitles]
    # merge the numbered and unnumbered sorting
    return nfiles + afiles, ntitles + atitles


def get_notebooks(path):
    """ Gets a list of all notebooks in a directory """
    # get list of files and their titles
    try:
        files = [path + f for f in os.listdir(path) if f.endswith('.md')]
    except FileNotFoundError:
        return {}
    titles = [get_title(f) for f in files]
    # sort the files and titles for display
    files_sorted, titles_sorted = sort_files_titles(files, titles)
    # generate notebook objects from the sorted lists and return
    notebooks = [notebook(f, t) for f, t in zip(files_sorted, titles_sorted)]
    return notebooks


tutorials = {}
# get tutorials for different directories
for dir in tutorial_directories:
    tutorials[dir] = get_notebooks(directory + dir + '/')

# Load environment for Jinja and template
env = Environment(
    loader=FileSystemLoader("./"),
    autoescape=select_autoescape()
)
template = env.get_template("index_template.html.jinja")

# render template and store
html = template.render(tutorials=tutorials)
with open('index.html', 'w+') as f:
    f.write(html)
