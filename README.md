## Team Mental Models for Human-Robot Interaction

In this project we aim to enable a robot to estimate the belief state of other (human) agents in its surroundings. We limit our work to a household domain, where a human completes general taskwork and the robot constructs a model to match what the user believes.

### Setup

We are exploring two simulation platforms for this task, Habitat-sim and VirtualHome. The simulators have the following tradeoffs:

Photo realism: Habitat
Physics realism: Habitat
Robot controller realism: Habitat
Dataset of human navigation in the household: Virtualhome
Execution of high-level plans: Virtualhome (out-of-the-box)
Simulation speed: Habitat
Ease of development: Virtualhome

We chose to go with VirtualHome for ease of development, however may later move to Habitat for a future project.

### Requirements

- VirtualHome, place in your project workspace, following the instructions on their GitHub
- PyTorch
- HRNet
- HYDRA-ROS

### Running the simulator (VirtualHome)

The project is slightly modified from the VirtualHome API, and includes a few scripts to run things. We developed this using Ubuntu 20.04.

To launch the simulation setup:

`python3 launch_sim.py`

This will launch the simulation for an example household, and spawn two agents.

`python3 run_agent_both_simultaneous.py`

This will start moving the agents around as they relocate 10 random objects around the household, and will record/save their videos.

Note: if the agent scripts ends prematurely and it appears to be because of a timeout, try decreasing the image resolution, decreasing the framerate, or increasing the processing time limit, all in `render_script(...)`.

The video frames will be located in `./Output/human/{CHAR}`. Each frame has an RGB image and a depth image, and the folder contans a `pd_human.txt` file (pose data for each frame) and a script `ftaa_human.txt` of the character's actions.

We have an example output here: https://drive.google.com/file/d/1ykeqP9-2GsfB0wvM3vzWyle9P7ZRXlhK/view?usp=sharing

Since we were unable to get VirtualHome to utilize our GPU, we recorded the simulation data and then relayed it to Hydra using ROS.

To relay the data from the `./Output` folder to ROS, run:

`python3 ros_interface.py`

This relays the RGB camera, Depth camera, and Pose (location/orientation) of the robot agent for each recorded frame. The orientation is calculated using the vector normal to the plane defined by the hip location and each shoulder location.

To run the example uhumans2 ROS bag, run:

`rosbag play /home/kolb/GT/sample_data/uHumans2_office_s1_00h.bag --clock`

I found the only topics needed are:
- /tesse/depth_cam/mono/image_raw
- /tesse/left_cam/rgb/image_raw
- /tesse/seg_cam/camera_info
- /tesse/seg_cam/rgb/image_raw
- /tf

Which can be run with:

`rosbag play uhumans2_nofrontlidar.bag --topics /tesse/depth_cam/mono/image_raw /tesse/left_cam/rgb/image_raw /tesse/seg_cam/camera_info /tesse/seg_cam/rgb/image_raw /tf`

### Scene Graph

We use Hydra to construct a scene graph from the robot's perspective. This requires us to give the robot's depth camera (parsed from the simulator output), the robot's pose/orientation (used the hip as the x/y location and the normal from the hip+Lshoulder+Rshoulder plane as the forward direction), and a semantic map (used HRNet).

We downloaded an HRNET model and are using the ADE dataset because the Hydra paper uses that model too. It took quite a bit of work to get HRNet working on custom camera inputs.

I found that the launch file has an interesting relationship with the ground truth, to a point where I wonder if it is reliant on the ground truth data, as HYDRA silently failed to generate a scene graph when the `use_gt_semantics` switch was set to false. It looks like Hydra is in the middle of a quality-of-life upgrade (and is still reliant on some code locked behind MIT's internal GitHub), so we are going to make our own dynamic scene graph generator in the meantime.

### (old) Custom DSG: Segmentation (Detectron2)

In our effort to create our own DSG, we are using Detectron2 for the semantic segmentation. I was unable to get detectron working on my local computer or the GT compute cluster, for some arcane install errors, so I had to use a conda environment.

Installing Detectron2 was challenging as it appears reliant on an older ligcc version and does not support python versions greater than 3.9.

I was unable to get Detectron installed via pip or their prebuilts, so I used a conda environment with pytorch 1.10.0 and torchvision 0.11. I then installed the Detectron2 conda package.

Running the python example code (`test.py`) failed from an "undefined symbol: _ZNK2at6Tensor7reshapeEN3c108ArrayRefIlEE" error, which the maintainers write is either because detectron2 requires an older libgcc version and they don't seem to have any intentions to rebuild detectron2 for newer runtimes versions, or there is a pytorch/torchvision mismatch.

I uninstalled pytorch, torchvision, and detectron2 via conda, and reinstalled detectron2 (which loaded what I think were its own listed target pytorch and torchvision packages). While importing detectron2 now works, I now had an error with an incompatibility between pytorch (2.3.1) and torchvision (0.15, which supports pytorch 2.0.1). I downgraded conda's pytorch (`conda install pytorch==2.0.1`). This also downgraded detectron2 to a previous version, which in turn downgraded torchvision to 0.8.2, which again led to the same problem.

I then tried updating the torchvision version, but that led to a different undefined symbol (pytorch jit). In another attempt, I tried to install pytorch version 1.10.1 and torchvision 0.11, and then the detectron prebuilt, which seemed to work. I then tried again on Python 12, and got it working.

While the process was messy, I think these are the steps:
1. Create conda environment with pytorch 1.10 and torchvision 0.11

`conda create -n "dt"`
`conda activate dt`
`conda install python==3.12.2`
`conda install conda-forge::pytorch==1.10.1`
`conda install conda-forge::torchvision==0.11.1`

2. On the Detectron2 conda download page, go to Files and download the .conda with the target architecture, python version, and CUDA/CPU. I think the issue was the default was incorrect here.
3. conda install the .conda file

`conda install detectron2*.conda`

4. Should be good to go, if you run into import errors just install the requisite pip packages

`pip3 install python-opencv matplotlib ...`

While I was able to hesitantly get Detectron2 to work, it wasn't great (fixed classes) and I could not get the GPU version to work (at the time of writing, only available for CUDA 12.1 while my GPU had CUDA 12.2). Also, there are more modern options today.

### Custom DSG: Segmentation (OWL2 + SAM2)

I then tried a different pipeline of using OWL2 for object detection, and SAM2 for object segmentation. OWL2 allows for open vocabulary object detection which is useful for a household environment and outputs bounding boxes for the objects. SAM2 does not take in a class, only bounding boxes or known points/paths, returning the most likely intended object. From our test cases this pipeline of open vocab detection -> bounding boxes -> segmentation works very well, and to greater fidelity than SAM2. Additionally, the two models are maintained and recent.

I installed OWL2 and SAM2 using their GitHub guides, to little issue.

I did need to set a number of environment variables:
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
