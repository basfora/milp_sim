# milp_sim

## Installation Guide

This project was developed in Python 3.6. It uses the following Python libraries: datetime, sys, os, pickle, numpy, matplotlib, igraph, gurobipy.

Start by cloning this repository, 
```
git clone https://github.com/basfora/milp_sim.git
```

### Run install script

This will install all the other necessary Python libraries and add the folder to your Python path system variable. From inside the `milp_sim` folder, run on terminal:
```
chmod +x install_script.sh
./install_script.sh
``` 

### Citing this work

If you use this algorithm or code, please cite our [paper](https://ieeexplore.ieee.org/abstract/document/9366368?casa_token=hrXCiSLKUMYAAAAA:djMlZYwuKUHiNHB5i-aWCQWZy98jSx7v5Tc1DCwKmlK5FFEwfsQI1TVH1OQ3UsLVYBVFbpwaJA):

```
@article{shree2021exploiting,
  title={Exploiting Natural Language for Efficient Risk-Aware Multi-Robot SaR Planning},
  author={Shree, Vikram and Asfora, Beatriz and Zheng, Rachel and Hong, Samantha and Banfi, Jacopo and Campbell, Mark},
  journal={IEEE Robotics and Automation Letters},
  volume={6},
  number={2},
  pages={3152--3159},
  year={2021},
  publisher={IEEE}
}
```