""" Script for generating indexes of the notebook. """

import argparse
import pathlib
import re

from jinja2 import (
    Environment,
    FileSystemLoader,
    select_autoescape,
)


TUTORIAL_DIRECTORIES = [
    'heom',
    'lectures',
    'pulse-level-circuit-simulation',
    'python-introduction',
    'quantum-circuits',
    'time-evolution',
    'optimal-control'
    'visualization',
    'miscellaneous'
]


def atoi(text):
    return int(text) if text.isdigit() else text


def natural_keys(text):
    return [atoi(c) for c in re.split(r'(\d+)', text)]


class Notebook:
    """ Notebook object for use in rendering templates. """

    NBVIEWER_URL_PREFIX = "https://nbviewer.org/urls/qutip.org/qutip-tutorials/"

    def __init__(self, title, tutorial_folder, path):
        self.tutorial_folder = tutorial_folder
        self.web_folder = tutorial_folder.parent

        self.title = title

        self.web_md_path = path.relative_to(self.web_folder)
        self.web_ipynb_path = self.web_md_path.with_suffix(".ipynb")

        self.tutorial_md_path = path.relative_to(self.tutorial_folder)
        self.tutorial_ipynb_path = self.tutorial_md_path.with_suffix(".ipynb")

        self.nbviewer_url = self.NBVIEWER_URL_PREFIX + self.web_ipynb_path.as_posix()
        self.try_qutip_url = "./tutorials/" + self.tutorial_ipynb_path.as_posix()


def get_title(path):
    """ Reads the title from a markdown notebook """
    with path.open('r') as f:
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
    nfiles = [s for s in files if s.name[0].isdigit()]
    nfiles = sorted(nfiles, key=lambda s: natural_keys(s.name))
    ntitles = [titles[files.index(s)] for s in nfiles]

    # sort the files without numbering by the alphabetic order of the titles
    atitles = [titles[files.index(s)] for s in files if s not in nfiles]
    atitles = sorted(atitles, key=natural_keys)
    afiles = [files[titles.index(s)] for s in atitles]

    # merge the numbered and unnumbered sorting
    return nfiles + afiles, ntitles + atitles


def get_notebooks(tutorials_folder, subfolder):
    """ Gets a list of all notebooks in a directory """
    files = list((tutorials_folder / subfolder).glob("*.md"))
    titles = [get_title(f) for f in files]
    files_sorted, titles_sorted = sort_files_titles(files, titles)
    notebooks = [
        Notebook(title, tutorials_folder, path)
        for title, path in zip(titles_sorted, files_sorted)
    ]
    return notebooks


def get_tutorials(tutorials_folder, tutorial_directories):
    """ Return a dictionary of all tutorials for a particular version. """
    tutorials = {}
    for subfolder in tutorial_directories:
        tutorials[subfolder] = get_notebooks(tutorials_folder, subfolder)
    return tutorials


def render_template(template_path, **kw):
    """ Render a Jinja template """
    env = Environment(
        loader=FileSystemLoader(str(template_path.parent)),
        autoescape=select_autoescape(),
    )
    template = env.get_template(template_path.name)
    return template.render(**kw)


def parse_args():
    parser = argparse.ArgumentParser(
        description="""
            Generate indexes for tutorial notebooks.

            This script is used both by this repository to generate the indexes
            for the QuTiP tutorial website and by https://github.com/qutip/try-qutip/
            to generate the notebook indexes for the Try QuTiP site.
        """,
    )
    parser.add_argument(
        "qutip_version", choices=["v4", "v5"],
        metavar="QUTIP_VERSION",
        help="Which QuTiP version to generate the tutorial index for [v4, v5].",
    )
    parser.add_argument(
        "index_type", choices=["html", "try-qutip"],
        metavar="INDEX_TYPE",
        help=(
            "Whether to generate an HTML index for the website or"
            " a Markdown Jupyter notebook index for the Try QuTiP site"
            " [html, try-qutip]."
        ),
    )
    parser.add_argument(
        "output_file",
        metavar="OUTPUT_FILE",
        help="File to write the index to.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    root_folder = pathlib.Path(__file__).parent.parent

    if args.qutip_version == "v4":
        title = "Tutorials for QuTiP Version 4"
        tutorials_folder = root_folder / "tutorials-v4"
        version_note = """
            These are the tutorials for QuTiP Version 4. You can
            find the tutorials for QuTiP Version 5
            <a href="./index.html">here</a>.
        """.strip()
    elif args.qutip_version == "v5":
        title = "Tutorials for QuTiP Version 5"
        tutorials_folder = root_folder / "tutorials-v5"
        version_note = """
            These are the tutorials for QuTiP Version 5. You can
            find the tutorials for QuTiP Version 4
            <a href="./index-v4.html">here</a>.
        """.strip()
    else:
        raise ValueError(f"Unsupported qutip_version: {args.qutip_version!r}")

    tutorials = get_tutorials(tutorials_folder, TUTORIAL_DIRECTORIES)

    if args.index_type == "html":
        template = root_folder / "website" / "index.html.jinja"
        text = render_template(
            template,
            title=title,
            version_note=version_note,
            tutorials=tutorials,
        )
    elif args.index_type == "try-qutip":
        template = root_folder / "website" / "index.try-qutip.jinja"
        text = render_template(
            template,
            title=title,
            tutorials=tutorials,
        )
    else:
       raise ValueError(f"Unsupported index_type: {args.index_type!r}")

    with open(args.output_file, "w") as f:
        f.write(text)


if __name__ == "__main__":
    main()
