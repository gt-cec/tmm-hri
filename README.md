## Team Mental Models for Human-Robot Interaction

In this project we aim to enable a robot to estimate the belief state of other (human) agents in its surroundings. We use a household domain for our experiments.

### Installation

#### Requirements

- (If generating your own simulation) VirtualHome, place in your project workspace, following the instructions on their GitHub
- PyTorch
- MMPose (and by extension, MMEngine and MMCV)
- OWLv2
- SAM2
- numpy (at the time of writing, SAM2/PyTorch required numpy<2, we used 1.26.1)

If you run into issues while installing these models, check the cross compatability of your PyTorch and Python versions. That caused us a lot of headaches.

#### Mamba Environment

Because a recent update of setuptools broke many packages with Python 3.12+, we use Python 3.11 for this project. Therefore we recommend using a mamba environment as Python 3.11 is a few years old at this point.

`mamba env create -n tmm-hri python=3.11`

`mamba activate tmm-hri`

Then, install the usual suite:

`mamba install numpy==1.26.4 matplotlib opencv easydict transformers`

#### Detection Stack (OWL2 + SAM2)

We are using OWLv2 for object detection, and SAM2 for object segmentation. OWLv2 allows for open vocabulary object detection which is useful for a household environment and outputs bounding boxes for the objects. SAM2 does not take in a class, only bounding boxes or known points/paths, returning the likely intended segmentation. From our test cases this pipeline of open vocab detection -> bounding boxes -> segmentation works very well.

Install OWLv2 and SAM2 using their GitHub guides. Again, we used a mamba environment and Python 3.11 as some libraries did not support Python>3.11 at this time (Fall 2024). Alternatively, if you are using a Mamba environment you can install SAM2 from the `conda-forge` repository:

`mamba install conda-forge::sam-2`

After installing SAM2, download the [model weights `.pt` file](https://huggingface.co/facebook/sam2.1-hiera-large/tree/main) and place it into the `segmentation` folder. We used `sam2.1-hiera-large`.

If you are using CUDA, remember to set the CUDA environment variables in your `.bashrc`:
```
export CUDA=12.6
export CUDA_HOME=~/miniconda/envs/tmmhri
export PATH=$CUDA_HOME/bin${PATH:+:${PATH}}
export CUDA_PATH=$CUDA_HOME
export LIBRARY_PATH=$CUDA_HOME/nvvm/lib64:$LIBRARY_PATH
export LD_LIBRARY_PATH=$CUDA_HOME/nvvm/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
export NVCC=$CUDA_HOME/bin/nvcc
export CFLAGS="-I$CUDA_HOME/include $CFLAGS"
export CUDA_TOOLKIT_ROOT_DIR=$CUDA_HOME
export LD_LIBRARY_PATH=$CUDA_HOME/extras/CUPTI/lib64:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib:$LD_LIBRARY_PATH
```

(As of Oct. 2024) If you are using a Mac, MPS is not supported well by PyTorch so we need to enable the fallback, add the following export to your terminal instance or add it to your `~/.zshrc`:

`export PYTORCH_ENABLE_MPS_FALLBACK=1`

#### Pose Stack (MMPose + RTMPose)

We use the MMPose API with RTMPose to obtain 2D keypoints, and a pose lifter model we got from a colleague. Place the pose lifter model (in our case, `best_MPJPE_epoch_98.pth`) into `pose_estimation/models/`.

Use the following steps to install MMPose. MMCV requires 2.1.0 for no reason other than one of MMPose's pose detection models has not had its dependency list updated.

`cd pose_estimation`
`git clone https://github.com/open-mmlab/mmpose.git`
`cd mmpose`
`python3.11 -m pip install mmcv==2.1.0 mmdet`
`python3.11 -m pip install -r requirements.txt`
`python3.11 -m pip install -v -e .`

Now head back to the root directory:

`cd ../../`

#### Simulator (VirtualHome)

VirtualHome has two components: an executable that runs a Unity player for the simulation, and a Python API that is used to interact with the simulator. Download the simulator executable corresponding to your platform from the VirtualHome documentation [here](https://github.com/xavierpuigf/virtualhome/tree/master?tab=readme-ov-file#download-unity-simulator). Move the .zip file to the project directory and unzip it, it should make a subfolder called `linux_exec` containing the Unity player and an executable `tmm-hri/linux_exec/linux_exec_v2.3.0.x86_64`.

**NOTE:** On Linux you will need to make the executable actually runnable: `chmod +x ./linux_exec/linux_exec_v2.3.0.x86_64`, otherwise you will run into `Error 2 file or directory not found` and some memory dump information.

We had to make some minor modifications to the VirtualHome API for it to work, so we include a stripped-down fork of the API in this project repo. VirtualHome is scarcely maintained so this code should stay relevant for the foreseeable future.

To run the simulator, first launch the executable, and then open a terminal and run one of the toy problem scripts (see the simulation section).

### Generating a simulation episode

We evaluate on two toy problems: *Cleaning Up*, and *Parents Are Out*. They represent household scenarios in which our methods would be useful.

Generating an episode for the scenario is rather straightforward:

1. Launch the simulator executable you downloaded.

2. Run the Python script we made for the scenario.

Note: VirtualHome is prone to failing due to timeouts, we have found success by decreasing the image resolution, decreasing the framerate, or increasing the processing time limit, all done in the `render_script(...)` function call. In general. do not expect more than ~10-30 pick and places before it crashes.

The video frames will be located in `./Output/human/{CHAR}`. Each frame has an RGB image and a depth image, and the folder contans a `pd_human.txt` file (pose data for each frame) and a script `ftaa_human.txt` of the character's actions.

**NOTE:** VirtualHome has no mechanism for getting the class labels of the `seg_class` output images, which must be reconstructed using the `seg_inst` images and the known object ID -> instance color map from the simulator. HOWEVER, the instance color map is not 1:1 -- multiple object classes can and do share an instance color. In addition, classes can have multiple colors in the `seg_class` images. Therefore some guesswork must be done to figure out the class colors. The first time an episode is run, a function `load_colormap()` goes through all the frames to try and match the instances to class colors. Where conflicts exist, the user (you) is asked to resolve them. Respond to the text prompt with the correct class, and press any key on the window that pops up. For better results, handcraft the correspondance as a dictionary in a file called `handcrafted_colormap.txt` in the episode's directory. Unfortunately this is just a limitation of VirtualHome. We found the class color map to be 1:1.

#### Cleaning Up

Cleaning Up has two agents rearrange objects in the household over a long time duration. The changing environment allows for evaluation on maintaining estimates of belief states over time. Agents follow a state machine where they simultaneously go to a random object, pick it up, and place it on a different surface. Due to limitations of VirtualHome, we were unable to get episodes to last longer than ~30 pick/place cycles, or around 20 minutes of runtime.

After launching the simulator, generate an episode by running:

`python3 sim_pick_and_place.py`

This will start moving the agents around as they relocate random objects around the household, and will record/save their camera feeds.

#### Parents Are Out

Parents Are Out aims to create a disparity between the human agent's belief state and the ground truth, such that the disparity can be leveraged for the downstream task of repairing the belief state. Agents are first exposed to a "clean" household where the objects are in their nominal locations. Objects are then randomly rearranged, and the human agent (with the robot following) briefly walk through the house. This episode is used for estimating the human's belief state such that the robot can later inform the human of objects they are not aware of.

After launching the simulator, generate an episode by running:

`python3 sim_rearrangement_walkthrough.py`

This will start moving the agents to follow the initialization and walkthrough, and will record/save their camera feeds.

### Preprocessing a simulation episode

The privileged information given in the episodes can be preprocessed for faster loading later on. Run:

`python3 preprocess_sim_detection.py`

The script pulls the ground truth object detection, segmentation masks, pose information, and projects the objects into their 3D locations. This is helpful for downstream ablation studies or model training where those operations are computationally expensive. The outputs are stored in `.pkl` files

Running the dynamic scene graph tests on privileged information uses the preprocessed `.pkl` files to save time.

### Constructing a DSG from a simulation episode

To run the dynamic scene graph, run:

`python3.11 test_dsg_sim.py`

This will open each pkl file for an agent and run the DSG on it, outputting the result plots to `visualization_frames/`.

You can convert the frames to a video using `frames_to_vid.sh`, which will save to `output.mp4`.

### Replaying a simulation episode via ROS

For testing the ROS system, we wrote a script to relay data from the `./episodes` folder to ROS. Run:

`python3 ros_interface.py`

This relays the RGB camera, Depth camera, and Pose (location/orientation) of the robot agent for each recorded frame. The orientation is calculated using the vector normal to the plane defined by the hip location and each shoulder location.