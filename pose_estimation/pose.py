# pose.py: conducts 2D pose detection and lifting to 3D

import utils
import pose_estimation.plots
import pose_estimation.hack_registry
import cv2
import numpy as np
from mmpose.apis import (_track_by_iou,
                         convert_keypoint_definition, extract_pose_sequence,
                         inference_pose_lifter_model, inference_topdown,
                         init_model)
from mmpose.structures import (PoseDataSample, merge_data_samples)
import os, torch, sys

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"  # required for OpenCV to load .exr files (depth)

# set device to cuda if cuda is available
if torch.cuda.is_available():
    device = "cuda"
# otherwise check if on macos
elif sys.platform == "darwin":
    device = "mps"
else:
    device = "cpu"

"""
Processing depth but stored as rgba.
"""

keypoint_index = {
    'root': 0, 'right_hip': 1, 'right_knee': 2, 'right_foot': 3,
    'left_hip': 4, 'left_knee': 5, 'left_foot': 6, 'spine': 7,
    'thorax': 8, 'neck_base': 9, 'head': 10, 'left_shoulder': 11,
    'left_elbow': 12, 'left_wrist': 13, 'right_shoulder': 14,
    'right_elbow': 15, 'right_wrist': 16
}

skeleton_links = [
    ('root', 'left_hip'), ('left_hip', 'left_knee'), ('left_knee', 'left_foot'),
    ('root', 'right_hip'), ('right_hip', 'right_knee'), ('right_knee', 'right_foot'),
    ('root', 'spine'), ('spine', 'thorax'), ('thorax', 'neck_base'),
    ('neck_base', 'head'), ('thorax', 'left_shoulder'), ('left_shoulder', 'left_elbow'),
    ('left_elbow', 'left_wrist'), ('thorax', 'right_shoulder'), ('right_shoulder', 'right_elbow'),
    ('right_elbow', 'right_wrist')
]

class PoseDetection:
    def __init__(self):
        print("Initialize pose detector")
        self.pose_estimator_config = './pose_estimation/models/rtmpose-m_8xb256-420e_body8-256x192.py'
        self.pose_estimator_checkpoint = 'https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-m_simcc-body7_pt-body7_420e-256x192-e48f03d0_20230504.pth'  # Pose estimator checkpoint file path
        self.pose_lifter_config = './pose_estimation/models/video-pose-lift_tcn-27frm-supv_8xb128-160e_fit3d.py'  # Pose lifter configuration file path
        self.pose_lifter_checkpoint = './pose_estimation/models/best_MPJPE_epoch_98.pth'
        self.device = device  # Device to use (e.g., 'cuda' or 'cpu')
        self.det_cat_id = 0  # Category ID for detection (e.g., person)
        self.bbox_thr = 0.5  # Bounding box threshold
        self.tracking_thr = 0.3  # Tracking threshold
        self.disable_norm_pose_2d = True  # Disable 2D pose normalization
        self.disable_rebase_keypoint = False  # Disable keypoint rebase
        self.kpt_thr = 0.5  # Keypoint threshold for visualization
        self.num_instances = -1  # Number of instances to visualize
        self.show = False  # Whether to show visualization
        self.show_interval = 0  # Interval between frames for visualization
        self.pose_estimator = init_model(self.pose_estimator_config, self.pose_estimator_checkpoint, device=self.device.lower())
        self.pose_lifter = init_model(self.pose_lifter_config, self.pose_lifter_checkpoint, device=self.device.lower())
        print("Completed initializing pose detector")

    # runs pose estimation on an RGB image given bounding boxes
    def process_one_image(self, args, robot_rgb, bounding_boxes, visualize_frame=False):
        """Visualize detected and predicted keypoints of one image."""
        pose_est_results_last = []
        # estimate pose results for current image
        if isinstance(bounding_boxes, list):
            bounding_boxes = np.array(bounding_boxes)
        flattened_bounding_boxes = np.zeros((0, 4))
        for i in range(bounding_boxes.shape[0]):
            flattened_bounding_boxes = np.append(flattened_bounding_boxes, np.array([[bounding_boxes[i,0,1], bounding_boxes[i,0,0], bounding_boxes[i,1,1], bounding_boxes[i,1,0]]]), axis=0)
        bounding_boxes = flattened_bounding_boxes[0]  # limited to one bounding box per frame at the moment for some reason
        pose_est_results = inference_topdown(self.pose_estimator, robot_rgb, [bounding_boxes])
        _track = _track_by_iou

        pose_det_dataset_name = self.pose_estimator.dataset_meta['dataset_name']
        pose_est_results_converted = []

        # convert 2d pose estimation results into the format for pose-lifting
        # such as changing the keypoint order, flipping the keypoint, etc.
        next_id = 0
        for i, data_sample in enumerate(pose_est_results):
            pred_instances = data_sample.pred_instances.cpu().numpy()
            keypoints = pred_instances.keypoints
            # calculate area and bbox
            if 'bboxes' in pred_instances:
                areas = np.array([(bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
                                for bbox in pred_instances.bboxes])
                pose_est_results[i].pred_instances.set_field(areas, 'areas')
            else:
                areas, bboxes = [], []
                for keypoint in keypoints:
                    xmin = np.min(keypoint[:, 0][keypoint[:, 0] > 0], initial=1e10)
                    xmax = np.max(keypoint[:, 0])
                    ymin = np.min(keypoint[:, 1][keypoint[:, 1] > 0], initial=1e10)
                    ymax = np.max(keypoint[:, 1])
                    areas.append((xmax - xmin) * (ymax - ymin))
                    bboxes.append([xmin, ymin, xmax, ymax])
                pose_est_results[i].pred_instances.areas = np.array(areas)
                pose_est_results[i].pred_instances.bboxes = np.array(bboxes)

            # track id
            track_id, pose_est_results_last, _ = _track(data_sample, pose_est_results_last, args.tracking_thr)
            if track_id == -1:
                if np.count_nonzero(keypoints[:, :, 1]) >= 3:
                    track_id = next_id
                    next_id += 1
                else:
                    # If the number of keypoints detected is small,
                    # delete that person instance.
                    keypoints[:, :, 1] = -10
                    pose_est_results[i].pred_instances.set_field(keypoints, 'keypoints')
                    pose_est_results[i].pred_instances.set_field(pred_instances.bboxes * 0, 'bboxes')
                    pose_est_results[i].set_field(pred_instances, 'pred_instances')
                    track_id = -1
            pose_est_results[i].set_field(track_id, 'track_id')

            # convert keypoints for pose-lifting
            pose_est_result_converted = PoseDataSample()
            pose_est_result_converted.set_field(pose_est_results[i].pred_instances.clone(), 'pred_instances')
            pose_est_result_converted.set_field(pose_est_results[i].gt_instances.clone(), 'gt_instances')
            keypoints = convert_keypoint_definition(keypoints, pose_det_dataset_name, "h36m")
            pose_est_result_converted.pred_instances.set_field(keypoints, 'keypoints')
            pose_est_result_converted.set_field(pose_est_results[i].track_id, 'track_id')
            pose_est_results_converted.append(pose_est_result_converted)
        
        # Second stage: Pose lifting
        # extract and pad input pose2d sequence
        pose_est_results_list = [pose_est_results_converted.copy()]
        pose_seq_2d = extract_pose_sequence(pose_est_results_list, frame_idx=0, causal=False, seq_len=1, step=1)

        # conduct 2D-to-3D pose lifting
        norm_pose_2d = not self.disable_norm_pose_2d
        pose_lift_results = inference_pose_lifter_model(
            self.pose_lifter,
            pose_seq_2d,
            image_size=visualize_frame.shape[:2],
            norm_pose_2d=norm_pose_2d)

        # post-processing
        for idx, pose_lift_result in enumerate(pose_lift_results):
            pose_lift_result.track_id = pose_est_results[idx].get('track_id', 1e4)

            pred_instances = pose_lift_result.pred_instances
            keypoints = pred_instances.keypoints
            keypoint_scores = pred_instances.keypoint_scores
            if keypoint_scores.ndim == 3:
                keypoint_scores = np.squeeze(keypoint_scores, axis=1)
                pose_lift_results[
                    idx].pred_instances.keypoint_scores = keypoint_scores
            if keypoints.ndim == 4:
                keypoints = np.squeeze(keypoints, axis=1)

            keypoints = keypoints[..., [0, 2, 1]]
            keypoints[..., 2] = -keypoints[..., 2]

            # rebase height (z-axis)
            if not args.disable_rebase_keypoint:
                keypoints[..., 2] -= np.min(
                    keypoints[..., 2], axis=-1, keepdims=True)

            pose_lift_results[idx].pred_instances.keypoints = keypoints

        pose_lift_results = sorted(
            pose_lift_results, key=lambda x: x.get('track_id', 1e4))

        pred_3d_data_samples = merge_data_samples(pose_lift_results)
        pred_3d_instances = pred_3d_data_samples.get('pred_instances', None)

        if args.num_instances < 0:
            args.num_instances = len(pose_lift_results)

        return pose_est_results_list, pred_3d_instances
    
    def get_heading_of_person(self, rgb, depth_map, detected_humans, robot_pose):
        robot_loc = robot_pose[0]
        robot_heading = robot_pose[1]
        # extract the boxes
        seg_pixel_locations = detected_humans[0]["seg mask"]

        # get the pose lifting
        pred_2d_poses, pred_3d_instances = self.process_one_image(
            args=self,
            robot_rgb=rgb,
            bounding_boxes=[[[coordinate[1], coordinate[0]] for coordinate in o["box"]] for o in detected_humans],  # boxes from detector use x,y not row,col, change to row,col
            visualize_frame=rgb)

        keypoints = [pose.pred_instances.keypoints[0] for pose in pred_2d_poses[0]]  # List of keypoints for all persons
        first_person_3d = pred_3d_instances.keypoints[0]

        mean_depth = utils.compute_mean_depth(seg_pixel_locations, depth_map)
        mean_depth += 0.15 # account for thickness of body adjustment.

        pred_person_loc = robot_loc.copy()

        heading_xy = robot_heading
        # Compute rotation angle
        theta = np.arctan2(heading_xy[1], heading_xy[0])
        theta_deg = np.degrees(theta)
        cos_theta, sin_theta = np.cos(theta), np.sin(theta)
        x_offset = cos_theta * mean_depth
        y_offset = sin_theta * mean_depth

        pred_person_loc[0] += x_offset
        pred_person_loc[1] += y_offset

        FOV = 60  # Horizontal field of view in degrees
        IMG_RES = (512, 512)  # Resolution of the image

        rw_coordinates = utils.calculate_rw_coordinates(keypoints[0][0], mean_depth, FOV, IMG_RES)
        # Rotate robot heading CW by 90° to get perpendicular heading
        perpendicular_heading = np.array([heading_xy[1], -heading_xy[0]])

        # Break into sin/cos components and translate by rw_coordinates[0]
        x_offset_perpendicular = perpendicular_heading[0] * rw_coordinates[0]
        y_offset_perpendicular = perpendicular_heading[1] * rw_coordinates[0]

        pred_person_loc[0] += x_offset_perpendicular
        pred_person_loc[1] += y_offset_perpendicular
        pred_skeleton = first_person_3d.copy()

        """
        Here we calculate the angle between the default camera centric heading (0,1) and the GT robot heading.
        """
        default_y_axis = np.array([0, 1])

        # Predicted robot heading (assumes it's already normalized)
        pred_robot_heading = robot_heading[:2]  # cut off the z axis because people usually don't tilt their heads

        # Compute the angle (in radians) using the dot product
        dot_product = np.dot(default_y_axis, pred_robot_heading)
        angle = np.arccos(np.clip(dot_product, -1.0, 1.0))  # Clip for numerical stability

        # Determine CCW or CW using the sign of the cross product
        cross_product = np.cross(np.append(default_y_axis, 0), np.append(pred_robot_heading, 0))
        if cross_product[-1] < 0:  # Z-component indicates direction
            angle = -angle

        # Create a 2D rotation matrix for the calculated angle
        rotation_matrix = np.array([
            [np.cos(angle), -np.sin(angle), 0],
            [np.sin(angle), np.cos(angle), 0],
            [0, 0, 1]
        ])

        # Apply the rotation to the skeleton
        pred_skeleton = np.dot(first_person_3d, rotation_matrix.T)  # Rotate skeleton
        pred_skeleton[...,0] += pred_person_loc[0]
        pred_skeleton[...,1] += pred_person_loc[1]
        
        # Calculate the orientation using shoulders
        left_shoulder_idx = keypoint_index['left_shoulder']
        right_shoulder_idx = keypoint_index['right_shoulder']

        # Predicted heading using shoulders and hip
        hip_loc = pred_skeleton[0][:3]  # Extract hip location as root
        right_shoulder_loc = pred_skeleton[right_shoulder_idx][:3]
        left_shoulder_loc = pred_skeleton[left_shoulder_idx][:3]

        # Compute forward vector (same logic as GT)
        predicted_forward = -1 * np.cross(right_shoulder_loc - hip_loc, left_shoulder_loc - hip_loc)
        predicted_heading = predicted_forward / np.linalg.norm(predicted_forward)

        return pred_person_loc, predicted_heading, (rw_coordinates, mean_depth, first_person_3d, keypoints, pred_skeleton)


    # get the forward direction of a character
    # pose: list of keypoints
    def get_direction_from_pose(self, pose:list, use_gt_human_pose=False) -> list:
        # get direction from ground truth pose
        if use_gt_human_pose:
            print("  Using ground truth human pose!")
            return pose[-1]  # use the coordinate system (east, north, vertical)
        # get direction from observed pose
        else:
            return pose  # use the coordinate system (east, north, vertical)


if __name__ == "__main__":
    pose = PoseDetection()

    # Run above file discovery
    padded_numbers = [
        "0017", "0018", "0019", "0020", "0021", "0022", "0023",
        "0423", "0424", "0425", "0426", "0427", "0428", "0429",
        "0430", "0432"
    ]

    episode_dir = "./episodes/episode_2024-09-04-16-32_agents_2_run_19"
    episode_name = "episode_2024-09-04-16-32_agents_2_run_19"

    ROBOT_POSES_PATH = f"{episode_dir}/0/pd_{episode_name}.txt"
    HUMAN_POSES_PATH = f"{episode_dir}/1/pd_{episode_name}.txt"
    robot_poses = utils.get_agent_pose_per_frame(episode_dir, episode_name, '0')
    human_poses = utils.get_agent_pose_per_frame(episode_dir, episode_name, '1')

    for FRAME_INDEX in padded_numbers:
        FRAME_INDEX_NONPAD = str(int(FRAME_INDEX))

        # Perform pose inference
        rgb_path = f'{episode_dir}/0/Action_{FRAME_INDEX}_0_normal.png'  # Input file path (image or video)
        frame = cv2.imread(rgb_path)
        seg_map_path = f'{episode_dir}/0/Action_{FRAME_INDEX}_0_seg_class.png'
        depth_map_path = f'{episode_dir}/0/Action_{FRAME_INDEX}_0_depth.exr'
        seg_map = cv2.imread(seg_map_path)
        depth_map = cv2.imread(depth_map_path,  cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)  # exr comes in at HxWx3, we want HxW

        detected_humans = []
        target_value = [124, 62, 0]  # RGB values for the target class
        seg_pixel_locations = np.where(np.all(seg_map == target_value, axis=-1))
        row_min = np.min(seg_pixel_locations[0])
        row_max = np.max(seg_pixel_locations[0])
        col_min = np.min(seg_pixel_locations[1])
        col_max = np.max(seg_pixel_locations[1])

        detected_humans.append({
            "seg mask": seg_pixel_locations,
            "box": np.array([[col_min, row_min], [col_max, row_max]])
        })

        robot_location = robot_poses[FRAME_INDEX_NONPAD][0]
        gt_person_loc = human_poses[FRAME_INDEX_NONPAD][0]
        gt_person_heading = human_poses[FRAME_INDEX_NONPAD][-1][:2] / np.linalg.norm(human_poses[FRAME_INDEX_NONPAD][-1][:2])
        robot_heading = robot_poses[FRAME_INDEX_NONPAD][-1][:2] / np.linalg.norm(robot_poses[FRAME_INDEX_NONPAD][-1][:2])

        pred_person_location, predicted_heading, (pred_human_relative_loc, mean_depth, first_person_3d, keypoints, pred_skeleton) = pose.get_heading_of_person(frame, depth_map, detected_humans, (robot_location, robot_heading))

        print("Frame", FRAME_INDEX, "Pred person loc", pred_person_location, "heading", predicted_heading)

        # if the ground truth data was supplied, visualize the heading
        if robot_location is not None and robot_heading is not None:
            pose_estimation.plots.visualize_combined(
                image=frame,  # Input image for 2D visualization
                first_person_3d = first_person_3d,  # 
                keypoints_2d=keypoints[0],  # 2D keypoints for the person
                skeleton_links=skeleton_links,  # Skeleton links for visualization
                keypoint_index=keypoint_index,  # Keypoint index mapping
                keypoints_3d=pred_skeleton,  # Predicted 3D skeleton
                robot_location=robot_location,  # Ground truth robot location
                robot_heading=robot_heading,  # Ground truth robot heading
                pred_person_location=pred_person_location,  # Predicted person location
                predicted_heading=predicted_heading,  # Predicted person heading
                pred_human_relative_loc=pred_human_relative_loc,  # Real-world coordinates of predicted person
                mean_depth=mean_depth,  # Mean root depth of predicted person
                bboxes=[o["box"] for o in detected_humans],
                gt_person_location=gt_person_loc,  # Ground truth person location
                gt_person_heading=gt_person_heading,  # Ground truth person heading
                save_path=f'./11_19_top_down_comb_{FRAME_INDEX}.png'
            )

        # ipdb.set_trace()
        print(f"Plot saved successfully for {FRAME_INDEX}")
