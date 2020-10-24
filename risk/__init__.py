import sys
import os

# add this folder to Python path
this_path = os.path.dirname(os.path.abspath(__file__))
my_path = sys.path
if this_path not in my_path:
    sys.path.insert(1, this_path)

# add milp_mespp to the Python path
parent_folder = os.path.dirname(this_path)
milp_folder = parent_folder + '/' + 'milp_mespp'

if milp_folder not in my_path:
    sys.path.insert(1, milp_folder)



