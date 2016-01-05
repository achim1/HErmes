sphinx-apidoc -f -o docs/apidocs pyevsel
python setup.py build_sphinx
touch docs/build/html/.nojekyll
rm -rf docs/build/doctrees
rm docs/build/Gemfile
echo "source 'https://rubygems.org'" > docs/build/Gemfile
echo "gem 'github-pages'" >> docs/build/Gemfile
mv docs/build/html docs/build/apidocs
cp -r docs/jekyll-docs/* docs/build
 
