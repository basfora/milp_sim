#!/bin/bash -i

# get milp_mespp path
MYSCRIPT=$(readlink -f "$0")
SCRIPTPATH=$(dirname "$MYSCRIPT")
# add to python path
echo '# milp_sim path' >> ~/.bashrc
echo export PYTHONPATH=\"\${PYTHONPATH}:$SCRIPTPATH\" >> ~/.bashrc
source ~/.bashrc
