#sphinx-apidoc -f -o docs/apidocs HErmes
sphinx-apidoc -f -o docs HErmes
python setup.py build_sphinx 
#touch docs/build/html/.nojekyll
#rm -rf docs/build/doctrees
#rm docs/build/Gemfile
#echo "source 'https://rubygems.org'" > docs/build/Gemfile
#echo "gem 'github-pages'" >> docs/build/Gemfile
#rm -rf docs/build/apidocs
#mkdir docs/build/apidocs
#cp -r docs/build/html/* docs/build/apidocs
##rm docs/apidocs/*
##mv docs/build/html/* docs/apidocs
##rmdir docs/build/html
#cp -r docs/jekyll-docs/* docs/build
# 
