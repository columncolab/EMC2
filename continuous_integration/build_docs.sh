set -e

echo "Building Docs"
conda install -c conda-forge -q sphinx doctr pandoc
conda install numpydoc 
pip install sphinx_gallery
pip install sphinx-copybutton
conda install -c conda-forge nbsphinx=0.8.3=pyhd8ed1ab_0
cd doc
make clean
make html
cd ..
doctr deploy . 
