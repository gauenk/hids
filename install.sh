#!/bin/bash

cd lib/
python setup.py clean --all
cd ..
python -m pip install -e ./lib --user
