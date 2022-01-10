#!/bin/bash

cd src/
python setup.py clean --all
cd ..
python -m pip install -e ./src --user
