import os
import re
from jinja2 import Environment, FileSystemLoader, select_autoescape


def atoi(text):
    return int(text) if text.isdigit() else text


def natural_keys(text):
    return [atoi(c) for c in re.split('(\d+)', text)]


class notebook:
    def __init__(self, path, title):
        # remove ../ from path
        self.path = path.replace('../', '')
        self.title = title
        # set url and update from markdown to ipynb
        self.url = url_prefix + self.path.replace(".md", ".ipynb")


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


def generate_index_html(version_directory, tutorial_directories, title,
                        version_note):
    """ Generates the index html file from the given data"""
    # get tutorials from the different directories
    tutorials = {}
    for dir in tutorial_directories:
        tutorials[dir] = get_notebooks(version_directory + dir + '/')

    # Load environment for Jinja and template
    env = Environment(
        loader=FileSystemLoader("../"),
        autoescape=select_autoescape()
    )
    template = env.get_template("website/qutip-qip.html.jinja")

    # render template and return
    html = template.render(tutorials=tutorials, title=title,
                           version_note=version_note)
    return html


# url prefix for the links
url_prefix = "https://nbviewer.org/urls/qutip.org/qutip-tutorials/"
# tutorial directories

qutip_qip_tutorial_directories = [
    'pulse-level-circuit-simulation',
    'quantum-circuits',
]
# +++ READ PREFIX AND SUFFIX +++
prefix = ""
suffix = ""

with open('prefix.html', 'r') as f:
    prefix = f.read()
with open('suffix.html', 'r') as f:
    suffix = f.read()

# +++ VERSION 4 INDEX FILE +++
title = 'Tutorials for QuTiP Version 4'
version_note = 'This are the tutorials for QuTiP Version 4. You can \
         find the tutorials for QuTiP Version 5 \
          <a href="./index-v5.html">here</a>.'

qutip_qip_html = generate_index_html('../tutorials-v4/', qutip_qip_tutorial_directories, title,
                           version_note)
with open('qutip-qip.html', 'w+') as f:
    f.write(prefix)
    f.write(qutip_qip_html)
    f.write(suffix)

# +++ VERSION 5 INDEX FILE +++
title = 'Tutorials for QuTiP Version 5'
version_note = 'This are the tutorials for QuTiP-QIP Version 5. You can \
         find the tutorials for QuTiP Version 4 \
          <a href="./index.html">here</a>.'

qutip_qip_html = generate_index_html('../tutorials-v5/', qutip_qip_tutorial_directories, title,
                           version_note)
with open('qutip-qip-v5.html', 'w+') as f:
    f.write(prefix)
    f.write(qutip_qip_html)
    f.write(suffix)
