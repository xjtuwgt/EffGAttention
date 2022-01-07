import sys
import os
from os.path import join

# Add submodule path into import paths
PROJECT_FOLDER = os.path.dirname(__file__)
print('Project folder = {}'.format(PROJECT_FOLDER))
sys.path.append(join(PROJECT_FOLDER))
# Define the dataset folder and model folder based on environment
HOME_DATA_FOLDER = join(PROJECT_FOLDER, 'data')
os.makedirs(HOME_DATA_FOLDER, exist_ok=True)
print('*' * 35, ' path information ', '*' * 35)
print('Data folder = {}'.format(HOME_DATA_FOLDER))
OUTPUT_FOLDER = join(HOME_DATA_FOLDER, 'output')
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
