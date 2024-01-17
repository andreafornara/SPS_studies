wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
bash Miniforge3-Linux-x86_64.sh -b  -p ./miniforge -f
source miniforge/bin/activate                                                    
python -m pip install -r requirements.txt
conda install mamba -n base -c conda-forge
pip install cupy-cuda11x
mamba install cudatoolkit=11.8.0
pip install gpustat     