curl -L -O "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"
bash Miniforge3-$(uname)-$(uname -m).sh -b  -p ./miniforge -f
source miniforge/bin/activate                                                    
python -m pip install -r requirements.txt
conda install mamba -n base -c conda-forge
pip install cupy-cuda11x
mamba install cudatoolkit=11.8.0
pip install gpustat