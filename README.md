# SPS_studies
Emittance Growth with Crab Cavity+Noise+Wakefields on a MAD-X lattice via XSuite

First of all create a working Miniforge via:
```
source install_miniforge.sh
```
Warning: Conda and cupy need to be modified accordingly to your own needs, the default case used here is working on the pcbe-abp-gpu001 .
Once the Python environment is created and activated launch the simulation by:
```
source launch_simulation.sh
```
To modify the parameters of the simulation use the config.yaml.