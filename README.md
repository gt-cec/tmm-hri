## Team Mental Models for Human-Robot Interaction

In this project we aim to enable a robot to estimate the belief state of other (human) agents in its surroundings. We use a household domain for our experiments.

### Installation

#### Requirements

- (If generating your own simulation) VirtualHome, place in your project workspace, following the instructions on their GitHub
- PyTorch
- MMPose
- OWLv2
- SAM2
- numpy (at the time of writing, SAM2/PyTorch required numpy<2, we used 1.26.1)

If you run into issues while installing these models, check the cross compatability of your PyTorch and Python versions. That caused us a lot of headaches.

#### Detection Stack (OWL2 + SAM2)

We are using OWLv2 for object detection, and SAM2 for object segmentation. OWLv2 allows for open vocabulary object detection which is useful for a household environment and outputs bounding boxes for the objects. SAM2 does not take in a class, only bounding boxes or known points/paths, returning the likely intended segmentation. From our test cases this pipeline of open vocab detection -> bounding boxes -> segmentation works very well.

Install OWLv2 and SAM2 using their GitHub guides. We used a conda environment and Python 3.11, as some libraries did not support Python>3.11 at this time (Fall 2024).

If you are using CUDA, remember to set the CUDA environment variables in your `.bashrc`:
```
export CUDA=12.2
export CUDA_HOME=~/anaconda3/envs/sam2  # NOTE: replace this with your anaconda environment path, I named mine sam2 for no particular reason.
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

We use the MMPose API with RTMPose to obtain 2D keypoints, and a pose lifter model we got from a colleague. Place the pose lifter model (in our case, `best_MPJPE_epoch_98.pth`) into `pose_estimation/models/`. You can 

#### Simulator (VirtualHome)

Installing VirtualHome has two steps: download the Python API, and download the simulator executable. Let's start with the API.

The VirtualHome instructions recommend using pip, however we found that the method does not work well. Instead, we can clone the VirtualHome repository into the project folder and access the API that way. Much easier, but use pip if you plan to install VirtualHome for multiple projects.

`git clone https://github.com/xavierpuigf/virtualhome`

(As of December 13, 2024) We submitted a bug fix that has not yet been accepted. Open `virtualhome/virtualhome/simulation/__init__.py` and replace the file contents with:

```
import glob
import sys
from sys import platform

# Needs to be fixed!
original_path = sys.path[5]
new_path = original_path + '/virtualhome/simulation'
sys.path.append(new_path)

# if installed via pip
try:
    from unity_simulator.comm_unity import UnityCommunication
    from unity_simulator import utils_viz

# if running locally (cloned into the project repository)
except ModuleNotFoundError:
    from .simulation.unity_simulator.comm_unity import UnityCommunication
    from .simulation.unity_simulator import utils_viz
```

Next, download the simulator executable of your choice from the VirtualHome documentation [here](https://github.com/xavierpuigf/virtualhome/tree/master?tab=readme-ov-file#download-unity-simulator). Place the executable anywhere.

To run the simulator, first launch the executable, and then open a terminal and run one of the toy problem scripts (see the other simulation section).


### Generating a simulation episode

We evaluate on two toy problems: *Cleaning Up*, and *Parents Are Out*. They represent household scenarios in which our methods would be useful.

Generating an episode for the scenario is rather straightforward:

1. Launch the simulator executable you downloaded.

2. Run the Python script we made for the scenario.

Note: VirtualHome is prone to failing due to timeouts, we have found success by decreasing the image resolution, decreasing the framerate, or increasing the processing time limit, all done in the `render_script(...)` function call. In general. do not expect more than ~10-30 pick and places before it crashes.

The video frames will be located in `./Output/human/{CHAR}`. Each frame has an RGB image and a depth image, and the folder contans a `pd_human.txt` file (pose data for each frame) and a script `ftaa_human.txt` of the character's actions.

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

`python3.12 test_dsg_sim.py`

This will open each pkl file for an agent and run the DSG on it, outputting the result plots to `visualization_frames/`.

You can convert the frames to a video using `frames_to_vid.sh`, which will save to `output.mp4`.

### Replaying a simulation episode via ROS

For testing the ROS system, we wrote a script to relay data from the `./episodes` folder to ROS. Run:

`python3 ros_interface.py`

This relays the RGB camera, Depth camera, and Pose (location/orientation) of the robot agent for each recorded frame. The orientation is calculated using the vector normal to the plane defined by the hip location and each shoulder location.