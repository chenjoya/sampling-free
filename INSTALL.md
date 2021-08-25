## Installation

### Requirements:
- GCC >= 4.9
- Anaconda (with python3)

### Option 1: Step-by-step installation

```bash
# first, make sure that your conda is setup properly with the right environment
# for that, check that `which conda`, `which pip` and `which python` points to the
# right path. From a clean conda env, this is what you need to do

conda create --name sampling-free python=3.8
conda activate sampling-free

# sampling-free and cocoapi dependencies
pip install ninja yacs cython matplotlib tqdm opencv-python pillow pycocotools

# follow PyTorch installation in https://pytorch.org/get-started/locally/
conda install pytorch torchvision cudatoolkit=11.1

# install sampling-free
git clone https://github.com/ChenJoya/sampling-free.git
cd sampling-free

# the following will install the lib with symbolic links, 
# so that you can modify the files if you want and won't need to re-build it
python setup.py build develop
