# installs the requisite packages using Mamba

# check if Mamba is installed
echo "Checking if Mamba is installed..."
if ! command -v mamba &> /dev/null
then
    echo "Mamba could not be found. Please install Mamba and try again."
    exit
fi

# if the tmm-hri environment is not present, create it
if ! mamba env list | grep -q "tmm-hri"
then
    echo "The tmm-hri environment does not exist. Creating it now..."
    mamba env create tmm-hri python=3.11
fi

# activate the tmm-hri environment
echo "Activating the tmm-hri environment..."
mamba activate tmm-hri

# install the mamba packages
echo "Installing the Mamba packages..."
mamba install numpy==1.26.4 matplotlib opencv easydict transformers conda-forge::sam-2

# install mmpose
echo "Installing mmpose..."
cd pose_detection
git clone https://github.com/open-mmlab/mmpose.git
cd mmpose
python3.11 -m pip install mmcv==2.1.0 mmdet
python3.11 -m pip install -v -e .`
cd ../../

# all done!
echo "Installation complete! Enjoy!"