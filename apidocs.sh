sphinx-apidoc -f -o docs/apidocs pyevsel
python setup.py build_sphinx
touch docs/html/.nojekyll

