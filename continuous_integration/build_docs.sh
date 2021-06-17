set -e

echo "Building Docs"
conda install -c conda-forge -q sphinx doctr pandoc
conda install numpydoc 
pip install sphinx_gallery
pip install sphinx-copybutton
conda install -c conda-forge nbsphinx
pip install Jinja2==2.11.3
cd doc
make clean
make html
cd ..
doctr deploy . 
