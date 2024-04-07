import os
import re
from jinja2 import Environment, FileSystemLoader, select_autoescape


def atoi(text):
    return int(text) if text.isdigit() else text


def natural_keys(text):
    return [atoi(c) for c in re.split('(\d+)', text)]


class Notebook:

    # url prefix for the links
    URL_PREFIX = "https://nbviewer.org/urls/qutip.org/qutip-tutorials/"

    def __init__(self, path, title):
        # remove ../ from path
        self.path = path.replace('../', '')
        self.title = title
        # set url and update from markdown to ipynb
        self.url = self.URL_PREFIX + self.path.replace(".md", ".ipynb")


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
    notebooks = [Notebook(f, t) for f, t in zip(files_sorted, titles_sorted)]
    return notebooks


def get_tutorials(version_directory, tutorial_directories):
    """ Return a dictionary of all tutorials for a particular version. """
    # get tutorials from the different directories
    tutorials = {}
    for dir in tutorial_directories:
        tutorials[dir] = get_notebooks(version_directory + dir + '/')
    return tutorials


def jinja_env():
    """ Return a Jinja environment for template rendering. """
    return Environment(
        loader=FileSystemLoader("../"),
        autoescape=select_autoescape(),
    )


def generate_index_html(title, version_note, tutorials):
    """ Generates the index html file from the given data. """
    env = jinja_env()
    template = env.get_template("website/index.html.jinja")
    html = template.render(tutorials=tutorials, title=title,
                           version_note=version_note)
    return html


def generate_index_notebook(title, tutorials):
    """ Generate an index Jupyter notebook in Markdown format. """
    env = jinja_env()
    template = env.get_template("website/index.md.jinja")
    md = template.render(tutorials=tutorials, title=title)
    return md


# tutorial directories
tutorial_directories = [
    'heom',
    'lectures',
    'pulse-level-circuit-simulation',
    'python-introduction',
    'quantum-circuits',
    'time-evolution',
    'visualization',
    'miscellaneous'
]

# +++ VERSION 4 INDEX FILE +++
title = 'Tutorials for QuTiP Version 4'
version_note = 'These are the tutorials for QuTiP Version 4. You can \
         find the tutorials for QuTiP Version 5 \
          <a href="./index.html">here</a>.'
tutorials_v4 = get_tutorials('../tutorials-v4/', tutorial_directories)

html = generate_index_html(title, version_note, tutorials_v4)
with open('index-v4.html', 'w+') as f:
    f.write(html)

notebook = generate_index_notebook(title, tutorials_v4)
with open('index.md', 'w+') as f:
    f.write(notebook)

# +++ VERSION 5 INDEX FILE +++
title = 'Tutorials for QuTiP Version 5'
version_note = 'These are the tutorials for QuTiP Version 5. You can \
         find the tutorials for QuTiP Version 4 \
          <a href="./index-v4.html">here</a>.'
tutorials_v5 = get_tutorials('../tutorials-v5/', tutorial_directories)

html = generate_index_html(title, version_note, tutorials_v5)
with open('index.html', 'w+') as f:
    f.write(html)

notebook = generate_index_notebook(title, tutorials_v5)
with open('index.md', 'w+') as f:
    f.write(notebook)
