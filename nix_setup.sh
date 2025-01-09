# installs the requisite packages using Mamba

echo "Welcome! This tool will create a Mamba environment and install the requisite packages for the TMM-HRI project."

# check if platform is Linux or MacOS
echo "Checking platform..."
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    echo "    Linux detected, moving on."
elif [[ "$OSTYPE" == "darwin"* ]]; then
    echo "    MacOS detected, moving on."
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
    mamba create -y -v -v -n tmm-hri python=3.11 numpy==1.26.4 matplotlib
else
    echo "    The tmm-hri environment already exists, remove it before running this script."
    exit
fi

# install mmpose
echo "Installing mmpose..."
cd pose_estimation
git clone https://github.com/open-mmlab/mmpose.git
cd mmpose
mamba run -n tmm-hri python3.11 -m pip install mmcv==2.1.0 mmdet
mamba run -n tmm-hri python3.11 -m pip install -v -e .`
cd ../../

# all done!
echo "Installation complete! Enjoy!"

# opencv easydict transformers conda-forge::sam-2