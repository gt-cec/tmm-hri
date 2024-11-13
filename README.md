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
- PyTorch (use the yolox)
- AlphaPose
- MotionBERT
- OwLv2
- SAM2

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

### Pose Detection

Niko recommended MotionBERT for 3D detection, which requires AlphaPose for 2D detection. I installed AlphaPose via their install guide, which required installing a few other packages. The only one that caused issues was halpecocotools, which could not be installed from `pip install halpecocotools` like the command line suggested, as I kept getting a `ModuleNotFound: 'numpy'` error. I was able to get it installed by git cloning the HalpeCOCOAPI repo and running `python setup.py build develop`. I fixed a quick error about `python not found` by changing halpecocotools' setup.py to use `python3` on the relevant cython line. Then, halpecocotools and AlphaPose installed without issue.

Due to cython using np.float and numpy having depreciated that, I added `np.float = float` to demo_inference.py so cython wouldn't break. Will probably need that in the real code too.

I had a circular import error with roi_align importing a cuda thing, and commented out that line. Similar issue with nms_cpu from detector.nms, which I was unable to resolve. It looks like an issue with their YOLOv3 detector, so tried using yolox instead. Yolox works. We should integrate this with Owlv2 once it works.

I integrated with OWLv2 and got AlphaPose working, also cleared out much of the unneeded code so now it's a fairly lightweight library.

Next is to use the AlphaPose keypoints for MotionBERT for 3D pose detection.

### Challenges with VirtualHome

For some reason, installing virtualhome ran into various problems with setuptools. I was instead able to modify the base package and import it as a submodule. I submitted pull requests to the main branch to support these changes.

### Running the dynamic scene graph

(optional) to pre-generate the agent visuals at each frame, run:

`./generate_pkl_processing.sh`

This will delete the existing pkl files for an agent (edit the .sh file to change the agent files to delete) and generate new ones for each frame.

To run the dynamic scene graph, run:

`python3.12 test_dsg_sim.py`

This will open each pkl file for an agent and run the DSG on it, outputting the result plots to `visualization_frames/`.
