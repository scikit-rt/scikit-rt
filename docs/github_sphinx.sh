#!/bin/bash

# Script for auto-generating Scikit-rt html documentation, using Sphinx.
# The script should be run from the Scikit-rt docs directory,
# in an environment where skrt can be imported, and where
# the following packages have been installed, e.g. via pip:
#     sphinx
#     sphinx-rtd-theme
#     myst-parser

# ------------------------------------------
# In setting up the document auto-generation, the following were useful:
#
# Step-by-step guide for getting started with sphinx:
# https://betterprogramming.pub/auto-documenting-a-python-project-using-sphinx-8878f9ddc6e9
# Sphinx documentation:
# https://www.sphinx-doc.org/en/master/
# Built-in themes:
# https://www.sphinx-doc.org/en/master/usage/theming.html#builtin-themes
# Third-party themes:
# https://sphinx-themes.org/
# ------------------------------------------

# Upgrade pip
python -m pip install --upgrade pip

# Workaraound for not being able to install mahotas from pypi
# grep -v mahotas ../setup.cfg > ./setup.cfg
# mv ./setup.cfg ../setup.cfg

# Workaround for having SimpleITK installation with Python 3.10
#pip install --upgrade --pre SimpleITK --find-links https://github.com/SimpleITK/SimpleITK/releases/tag/latest

# Install older version of jinja2 for compatibility with sphinx
#pin uninstall -y jinja2
#pip install jinja2==3.0.3

# Install scikit-rt, voxtox, and import
python -m pip install -e ..
python -m pip install -e ../examples/voxtox
python -m pip install -e ../examples/import
python -m pip install sphinx
python -m pip install sphinx-rtd-theme
python -m pip install myst-parser

# Delete package rst files, and recreate
EXCLUDE_PATTERN="../setup.py ../examples/voxtox/setup.py ../examples/import/setup.py"
rm -f source/skrt*.rst
rm -f source/voxtox*.rst
rm -f source/import*.rst
sphinx-apidoc -e -f --tocfile skrt_modules -o source ../src/skrt ${EXCLUDE_PATTERN}
sphinx-apidoc -e -f --tocfile voxtox_modules -o source ../examples/voxtox/src/voxtox ${EXCLUDE_PATTERN}
sphinx-apidoc -e -f --tocfile import_analysis_modules -o source ../examples/import/src/import_analysis ${EXCLUDE_PATTERN}

# Copy markdown files and images to be used in creating documentation.
rm -rf source/*.md
cp "../README.md" "source"
cp "image_registration.md" "source"
rm -rf "source/_static"
cp -rp "images" "source/_static"

# Change relative paths to linked files.
sed -i 's/docs\/image_registration/image_registration/' 'source/README.md'
sed -i 's/docs\/images/_static/' 'source/README.md'

# Delete and recreate html-format documentation
make clean
make html
