#!/bin/bash

# Generate rst files with autodoc directives from code.
# Only needed when a new module is added to the project,
# but does no harm to run every time.
sphinx-apidoc -o source ../src/skrt

# Copy markdown files and images to be used in creating documentation.
cp "../README.md" "source"
cp "image_registration.md" "source"
rm -rf "source/_static"
cp -rp "images" "source/_static"

# Change relative paths to linked files.
sed -i '' 's/docs\/image_registration/image_registration/' 'source/README.md'
sed -i '' 's/docs\/images/_static/' 'source/README.md'

# Delete and recreate html-format documentation
make clean
make html
