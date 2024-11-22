# Copyright (c) OpenMMLab. All rights reserved.
from functools import partial
import utils
import plots
import cv2
import numpy as np
from mmpose.apis import (_track_by_iou, _track_by_oks,
                         convert_keypoint_definition, extract_pose_sequence,
                         inference_pose_lifter_model, inference_topdown,
                         init_model)
from mmpose.structures import (PoseDataSample, merge_data_samples)
from mmpose.utils import adapt_mmdet_pipeline

try:
    from mmdet.apis import inference_detector, init_detector
    has_mmdet = True
except (ImportError, ModuleNotFoundError):
    has_mmdet = False

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
        self.det_config = './models/rtmdet_m_640-8xb32_coco-person.py'
        self.det_checkpoint = 'https://download.openmmlab.com/mmpose/v1/projects/rtmpose/rtmdet_m_8xb32-100e_coco-obj365-person-235e8209.pth'  # Detector checkpoint file path
        self.pose_estimator_config = './models/rtmpose-m_8xb256-420e_body8-256x192.py'
        self.pose_estimator_checkpoint = 'https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-m_simcc-body7_pt-body7_420e-256x192-e48f03d0_20230504.pth'  # Pose estimator checkpoint file path
        self.pose_lifter_config = './models/video-pose-lift_tcn-27frm-supv_8xb128-160e_fit3d.py'  # Pose lifter configuration file path
        self.pose_lifter_checkpoint = './models/best_MPJPE_epoch_98.pth'
        self.device = 'cpu'  # Device to use (e.g., 'cuda' or 'cpu')
        self.det_cat_id = 0  # Category ID for detection (e.g., person)
        self.bbox_thr = 0.5  # Bounding box threshold
        self.use_oks_tracking = False  # Whether to use OKS tracking
        self.tracking_thr = 0.3  # Tracking threshold
        self.disable_norm_pose_2d = True  # Disable 2D pose normalization
        self.disable_rebase_keypoint = False  # Disable keypoint rebase
        self.kpt_thr = 0.5  # Keypoint threshold for visualization
        self.num_instances = -1  # Number of instances to visualize
        self.show = False  # Whether to show visualization
        self.show_interval = 0  # Interval between frames for visualization
    
        self.detector = init_detector( self.det_config, self.det_checkpoint, device=self.device.lower())
        self.detector.cfg = adapt_mmdet_pipeline(self.detector.cfg)

        self.pose_estimator = init_model(self.pose_estimator_config, self.pose_estimator_checkpoint, device=self.device.lower())
        self.pose_lifter = init_model(self.pose_lifter_config, self.pose_lifter_checkpoint, device=self.device.lower())
        print("Completed initializing pose detector")

    # This script takes as input a single image, 
    # then outputs the mean depth of the person in the image
    # This equates to root depth, which we use to adjust the rootrel 3D pose estimation in test_demo.
    def process_one_image(self, args, detector, frame, frame_idx, pose_estimator,
                        pose_est_results_last, pose_est_results_list, next_id,
                        pose_lifter, visualize_frame, visualizer):
        """Visualize detected and predicted keypoints of one image.
        """
        pose_lift_dataset = pose_lifter.cfg.test_dataloader.dataset
        pose_lift_dataset_name = pose_lifter.dataset_meta['dataset_name']

        # First stage: conduct 2D pose detection in a Topdown manner
        # use detector to obtain person bounding boxes
        det_result = inference_detector(detector, frame)
        pred_instance = det_result.pred_instances.cpu().numpy()

        # filter out the person instances with category and bbox threshold
        # e.g. 0 for person in COCO
        bboxes = pred_instance.bboxes
        bboxes = bboxes[np.logical_and(pred_instance.labels == args.det_cat_id,
                                    pred_instance.scores > args.bbox_thr)]

        # estimate pose results for current image
        pose_est_results = inference_topdown(pose_estimator, frame, bboxes)

        if args.use_oks_tracking:
            _track = partial(_track_by_oks)
        else:
            _track = _track_by_iou

        pose_det_dataset_name = pose_estimator.dataset_meta['dataset_name']
        pose_est_results_converted = []

        # convert 2d pose estimation results into the format for pose-lifting
        # such as changing the keypoint order, flipping the keypoint, etc.
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
            track_id, pose_est_results_last, _ = _track(data_sample,
                                                        pose_est_results_last,
                                                        args.tracking_thr)
            if track_id == -1:
                if np.count_nonzero(keypoints[:, :, 1]) >= 3:
                    track_id = next_id
                    next_id += 1
                else:
                    # If the number of keypoints detected is small,
                    # delete that person instance.
                    keypoints[:, :, 1] = -10
                    pose_est_results[i].pred_instances.set_field(
                        keypoints, 'keypoints')
                    pose_est_results[i].pred_instances.set_field(
                        pred_instances.bboxes * 0, 'bboxes')
                    pose_est_results[i].set_field(pred_instances, 'pred_instances')
                    track_id = -1
            pose_est_results[i].set_field(track_id, 'track_id')

            # convert keypoints for pose-lifting
            pose_est_result_converted = PoseDataSample()
            pose_est_result_converted.set_field(
                pose_est_results[i].pred_instances.clone(), 'pred_instances')
            pose_est_result_converted.set_field(
                pose_est_results[i].gt_instances.clone(), 'gt_instances')
            # ipdb.set_trace()
            keypoints = convert_keypoint_definition(keypoints,
                                                    pose_det_dataset_name,
                                                    pose_lift_dataset_name)
            pose_est_result_converted.pred_instances.set_field(
                keypoints, 'keypoints')
            pose_est_result_converted.set_field(pose_est_results[i].track_id,
                                                'track_id')
            pose_est_results_converted.append(pose_est_result_converted)

        pose_est_results_list.append(pose_est_results_converted.copy())
        
        # ipdb.set_trace()
        # Second stage: Pose lifting
        # extract and pad input pose2d sequence
        pose_seq_2d = extract_pose_sequence(
            pose_est_results_list,
            frame_idx=frame_idx,
            causal=pose_lift_dataset.get('causal', False),
            seq_len=pose_lift_dataset.get('seq_len', 1),
            step=pose_lift_dataset.get('seq_step', 1))

        # conduct 2D-to-3D pose lifting
        norm_pose_2d = not self.disable_norm_pose_2d
        pose_lift_results = inference_pose_lifter_model(
            pose_lifter,
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
            # keypoints[..., 0] = -keypoints[..., 0]
            # keypoints[..., 1] = -keypoints[..., 1]
            keypoints[..., 2] = -keypoints[..., 2]

            # rebase height (z-axis)
            if not args.disable_rebase_keypoint:
                keypoints[..., 2] -= np.min(
                    keypoints[..., 2], axis=-1, keepdims=True)

            pose_lift_results[idx].pred_instances.keypoints = keypoints

        pose_lift_results = sorted(
            pose_lift_results, key=lambda x: x.get('track_id', 1e4))

        pred_3d_data_samples = merge_data_samples(pose_lift_results)
        det_data_sample = merge_data_samples(pose_est_results)
        pred_3d_instances = pred_3d_data_samples.get('pred_instances', None)

        if args.num_instances < 0:
            args.num_instances = len(pose_lift_results)

        # Visualization
        if visualizer is not None:
            visualizer.add_datasample(
                'result',
                visualize_frame,
                data_sample=pred_3d_data_samples,
                det_data_sample=det_data_sample,
                draw_gt=False,
                dataset_2d=pose_det_dataset_name,
                dataset_3d=pose_lift_dataset_name,
                show=args.show,
                draw_bbox=True,
                kpt_thr=args.kpt_thr,
                num_instances=args.num_instances,
                wait_time=args.show_interval)

        return pose_est_results, pose_est_results_list, pred_3d_instances, next_id
    
    def get_heading_of_person(self, frame, gt_person_loc=None, gt_person_heading=None, gt_robot_loc=None, gt_robot_heading=None):
        _, pred_2d_poses, pred_3d_instances, _ = self.process_one_image(
            args=self,
            detector=pose.detector,
            frame=frame,
            frame_idx=0,
            pose_estimator=pose.pose_estimator,
            pose_est_results_last=[],
            pose_est_results_list=[],
            next_id=0,
            pose_lifter=pose.pose_lifter,
            visualize_frame=frame,
            visualizer=None)

        keypoints = [pose.pred_instances.keypoints[0] for pose in pred_2d_poses[0]]  # List of keypoints for all persons
        first_person_3d = pred_3d_instances.keypoints[0]

        target_value = [124, 62, 0]  # RGB values for the target class

        mean_depth = utils.compute_mean_depth(seg_map_path, depth_map_path, target_value)
        mean_depth += 0.15 # account for thickness of body adjustment.

        PRED_PERSON_LOC= robot_poses[FRAME_INDEX_NONPAD][0].copy()

        heading_xy = GT_ROBOT_HEADING
        # Compute rotation angle
        theta = np.arctan2(heading_xy[1], heading_xy[0])
        theta_deg = np.degrees(theta)
        cos_theta, sin_theta = np.cos(theta), np.sin(theta)
        x_offset = cos_theta * mean_depth
        y_offset = sin_theta * mean_depth

        PRED_PERSON_LOC[0]+=x_offset
        PRED_PERSON_LOC[1]+=y_offset

        FOV = 60  # Horizontal field of view in degrees
        IMG_RES = (512, 512)  # Resolution of the image

        rw_coordinates = utils.calculate_rw_coordinates(keypoints[0][0], mean_depth, FOV, IMG_RES)
        # print("Real-world coordinates (X, Y, Z):", rw_coordinates)
        # Rotate robot heading CW by 90° to get perpendicular heading
        perpendicular_heading = np.array([heading_xy[1], -heading_xy[0]])

        # Break into sin/cos components and translate by rw_coordinates[0]
        x_offset_perpendicular = perpendicular_heading[0] * rw_coordinates[0]
        y_offset_perpendicular = perpendicular_heading[1] * rw_coordinates[0]

        PRED_PERSON_LOC[0] += x_offset_perpendicular
        PRED_PERSON_LOC[1] += y_offset_perpendicular
        PRED_SKELETON=first_person_3d.copy()

        """
        Here we calculate the angle between the default camera centric heading (0,1) and the GT robot heading.
        """
        default_y_axis = np.array([0, 1])

        # Predicted robot heading (assumes it's already normalized)
        pred_robot_heading = GT_ROBOT_HEADING

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
        PRED_SKELETON = np.dot(first_person_3d, rotation_matrix.T)  # Rotate skeleton
        PRED_SKELETON[...,0]+=PRED_PERSON_LOC[0]
        PRED_SKELETON[...,1]+=PRED_PERSON_LOC[1]
        
        # # Calculate the orientation using shoulders
        left_shoulder_idx = keypoint_index['left_shoulder']
        right_shoulder_idx = keypoint_index['right_shoulder']

        # Predicted heading using shoulders and hip
        hip_loc = PRED_SKELETON[0][:3]  # Extract hip location as root
        right_shoulder_loc = PRED_SKELETON[right_shoulder_idx][:3]
        left_shoulder_loc = PRED_SKELETON[left_shoulder_idx][:3]

        # Compute forward vector (same logic as GT)
        predicted_forward = -1 * np.cross(right_shoulder_loc - hip_loc, left_shoulder_loc - hip_loc)
        predicted_heading = predicted_forward / np.linalg.norm(predicted_forward)

        # if the ground truth data was supplied, visualize the heading
        if gt_person_loc is not None and gt_person_heading is not None and gt_robot_loc is not None and gt_robot_heading is not None:
            plots.visualize_combined(
                image=frame,  # Input image for 2D visualization
                first_person_3d = first_person_3d,  # 
                keypoints_2d=keypoints[0],  # 2D keypoints for the person
                skeleton_links=skeleton_links,  # Skeleton links for visualization
                keypoint_index=keypoint_index,  # Keypoint index mapping
                keypoints_3d=PRED_SKELETON,  # Predicted 3D skeleton
                GT_PERSON_LOC=gt_person_loc,  # Ground truth person location
                GT_PERSON_HEADING=gt_person_heading,  # Ground truth person heading
                GT_ROBOT_LOC=gt_robot_loc,  # Ground truth robot location
                GT_ROBOT_HEADING=gt_robot_heading,  # Ground truth robot heading
                PRED_PERSON_LOC=PRED_PERSON_LOC,  # Predicted person location
                predicted_heading=predicted_heading,  # Predicted person heading
                rw_coordinates=rw_coordinates,  # Real-world coordinates of predicted person
                mean_depth=mean_depth,  # Mean root depth of predicted person
                save_path=f'./11_19_top_down_comb_{FRAME_INDEX}.png'
            )

        return theta_deg


"""
Now that preliminary functions are done
"""

pose = PoseDetection()

if __name__ == "__main__":
    # Run above file discovery
    padded_numbers = [
        "0017", "0018", "0019", "0020", "0021", "0022", "0023",
        "0423", "0424", "0425", "0426", "0427", "0428", "0429",
        "0430", "0432"
    ]

    ROBOT_POSES_PATH = '../episodes/episode_2024-09-04-16-32_agents_2_run_19/0/pd_episode_2024-09-04-16-32_agents_2_run_19.txt'
    HUMAN_POSES_PATH = '../episodes/episode_2024-09-04-16-32_agents_2_run_19/1/pd_episode_2024-09-04-16-32_agents_2_run_19.txt'
    robot_poses = utils.get_agent_pose_per_frame(ROBOT_POSES_PATH)
    human_poses = utils.get_agent_pose_per_frame(HUMAN_POSES_PATH)

    for FRAME_INDEX in padded_numbers:
        FRAME_INDEX_NONPAD = str(int(FRAME_INDEX))

        # Perform pose inference
        rgb_path = f'../episodes/episode_2024-09-04-16-32_agents_2_run_19/0/Action_{FRAME_INDEX}_0_normal.png'  # Input file path (image or video)
        frame = cv2.cvtColor(cv2.imread(rgb_path), cv2.COLOR_BGR2RGB)
        seg_map_path = f'../episodes/episode_2024-09-04-16-32_agents_2_run_19/0/Action_{FRAME_INDEX}_0_seg_class.png'
        depth_map_path = f'../episodes/episode_2024-09-04-16-32_agents_2_run_19/0/Action_{FRAME_INDEX}_0_depth.exr'
        
        GT_PERSON_LOC = human_poses[FRAME_INDEX_NONPAD][0]
        GT_ROBOT_LOC = robot_poses[FRAME_INDEX_NONPAD][0]
        GT_PERSON_HEADING = human_poses[FRAME_INDEX_NONPAD][-1][:2] / np.linalg.norm(human_poses[FRAME_INDEX_NONPAD][-1][:2])
        GT_ROBOT_HEADING = robot_poses[FRAME_INDEX_NONPAD][-1][:2] / np.linalg.norm(robot_poses[FRAME_INDEX_NONPAD][-1][:2])

        theta_deg = pose.get_heading_of_person(frame, gt_person_loc=GT_PERSON_LOC, gt_person_heading=GT_PERSON_HEADING, gt_robot_loc=GT_ROBOT_LOC, gt_robot_heading=GT_ROBOT_HEADING)
        
        # ipdb.set_trace()
        print(f"Plot saved successfully for {FRAME_INDEX}")
        print("Degrees offset was: ", theta_deg)
