set -e

echo "Building Docs"
conda install -c conda-forge -q sphinx doctr pandoc
conda install numpydoc 
pip install sphinx_gallery
pip install sphinx-copybutton
conda install -c conda-forge nbsphinx
cd doc
make clean
make html
cd ..
doctr deploy . 
