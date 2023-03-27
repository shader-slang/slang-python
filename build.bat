python3 -m build --wheel --skip-existing --plat-name=win_amd64
pip install -e .
cd tests/
python -m unittest discover
cd ../