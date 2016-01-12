sphinx-apidoc -f -o docs/apidocs pyevsel
python setup.py build_sphinx
touch docs/build/html/.nojekyll
rm -rf docs/build/doctrees
rm docs/build/Gemfile
echo "source 'https://rubygems.org'" > docs/build/Gemfile
echo "gem 'github-pages'" >> docs/build/Gemfile
rm docs/build/apidocs/*
mv docs/build/html/* docs/build/apidocs
rmdir docs/build/html
cp -r docs/jekyll-docs/* docs/build
 
