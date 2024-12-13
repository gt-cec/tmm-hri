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

### Running the simulator (VirtualHome)

The project is slightly modified from the VirtualHome API, and includes a few scripts to help run everything. We developed this using Ubuntu 20.04.

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

### Running the dynamic scene graph

(optional) to pre-generate the agent visuals at each frame, run:

`./generate_pkl_processing.sh`

This will delete the existing pkl files for an agent (edit the .sh file to change the agent files to delete) and generate new ones for each frame.

To run the dynamic scene graph, run:

`python3.12 test_dsg_sim.py`

This will open each pkl file for an agent and run the DSG on it, outputting the result plots to `visualization_frames/`.

You can convert the frames to a video using `frames_to_vid.sh`, which will save to `output.mp4`.