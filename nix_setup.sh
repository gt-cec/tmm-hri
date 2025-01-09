# installs the requisite packages using Mamba

echo "Welcome! This tool will create a Mamba environment and install the requisite packages for the TMM-HRI project."

# check if platform is Linux or MacOS
echo "Checking platform..."
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    echo "    Linux detected."
elif [[ "$OSTYPE" == "darwin"* ]]; then
    echo "    MacOS detected."
else
    echo "!! Unsupported platform, this script is for a Linux or MacOS system. See the README for the full installation guide."
    exit
fi

# check if Mamba is installed
echo "Checking if Mamba is installed..."
if ! command -v mamba &> /dev/null
then
    echo "!! Mamba could not be found. Please install Mamba and try again."
    exit
else
    echo "    Mamba is installed, moving on."
fi

# if the tmm-hri environment is not present, create it
echo "Checking if the tmm-hri environment exists..."
if ! mamba env list | grep -q "tmm-hri"
then
    echo "    The tmm-hri environment does not exist. Creating it now..."
    mamba create -n tmm-hri python=3.11
else
    echo "    The tmm-hri environment already exists, moving on."
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