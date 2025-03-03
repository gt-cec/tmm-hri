import matplotlib.patches
import matplotlib.pyplot as plt
import cv2
import numpy as np

def visualize_combined(image, keypoints_2d, first_person_3d, skeleton_links, keypoint_index, keypoints_3d, 
                       robot_location, robot_heading, pred_person_location, predicted_heading, pred_human_relative_loc, mean_depth, bboxes, gt_person_location=None, gt_person_heading=None, save_path=''):
    """
    Combine 2D skeleton, 3D skeleton, and top-down view into a single figure.
    
    Args:
        image (np.ndarray): Input image (HxWx3).
        keypoints_2d (np.ndarray): Array of 2D keypoints (17x2 or 17x3 for scores).
        skeleton_links (list): List of tuples defining the skeleton links.
        keypoint_index (dict): Keypoint index mapping for skeleton links.
        keypoints_3d (np.ndarray): Array of 3D keypoints (17x3).
        GT_PERSON_LOC, GT_PERSON_HEADING, GT_ROBOT_LOC, GT_ROBOT_HEADING (tuple): Ground truth info.
        PRED_PERSON_LOC (tuple): Predicted person location.
        predicted_heading (np.ndarray): Heading vector calculated from shoulders.
        kpt_thr (float): Threshold for keypoint visibility.
        save_path (str): Path to save the combined visualization.
    """
    fig = plt.figure(figsize=(32, 8))

    # Plot 2D visualization
    ax1 = fig.add_subplot(1, 4, 1)
    ax1.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    for kp1, kp2 in skeleton_links:
        idx1, idx2 = keypoint_index[kp1], keypoint_index[kp2]
        x1, y1 = keypoints_2d[idx1][:2]
        x2, y2 = keypoints_2d[idx2][:2]
        ax1.plot([x1, x2], [y1, y2], color='blue', linewidth=2)
    for kp in keypoints_2d:
        x, y = kp[:2]
        ax1.scatter(x, y, color='red', s=50)
    ax1.set_title("2D Visualization")
    ax1.axis('off')
    for bbox in bboxes:
        ax1.add_patch(matplotlib.patches.Circle((bbox[0]), 5))
        ax1.add_patch(matplotlib.patches.Circle((bbox[1]), 5))
        rect = matplotlib.patches.Rectangle((bbox[0][0], bbox[0][1]), bbox[1][0]-bbox[0][0], bbox[1][1]-bbox[0][1], linewidth=1, edgecolor='r', facecolor='none')
        ax1.add_patch(rect)

    # Plot 3D skeleton
    ax2 = fig.add_subplot(1, 4, 2, projection='3d')
    ax2.view_init(elev=30, azim=60)
    for kp1, kp2 in skeleton_links:
        idx1, idx2 = keypoint_index[kp1], keypoint_index[kp2]
        x1, y1, z1 = keypoints_3d[idx1]
        x2, y2, z2 = keypoints_3d[idx2]
        ax2.plot([x1, x2], [y1, y2], [z1, z2], color='blue', linewidth=2)
    for kp in keypoints_3d:
        x, y, z = kp
        ax2.scatter(x, y, z, color='red', s=50)
    ax2.set_title("3D Skeleton")
    ax2.set_xlabel("X")
    ax2.set_ylabel("Y")
    ax2.set_zlabel("Z")
    ax2.set_xlim(-4, 4)
    ax2.set_ylim(-4, 4)
    ax2.set_zlim(0, 2)
    ax2.grid()

    # Plot top-down view
    all_x = []
    all_y = []
    
    ax3 = fig.add_subplot(1, 4, 3)
    if gt_person_location is not None:
        ax3.scatter(gt_person_location[0], gt_person_location[1], color="blue", label="GT Person", s=100, marker="o")
        ax3.arrow(gt_person_location[0], gt_person_location[1], 0.5 * gt_person_heading[0], 
            0.5 * gt_person_heading[1], head_width=0.2, head_length=0.3, fc="blue", ec="blue", label="GT Person Heading")
        all_x.append(gt_person_location[0])
        all_y.append(gt_person_location[1])
        
    ax3.scatter(robot_location[0], robot_location[1], color="green", label="GT Robot", s=100, marker="s")
    all_x.append(robot_location[0])
    all_y.append(robot_location[1])

    ax3.scatter(pred_person_location[0], pred_person_location[1], color="red", label="Predicted Person", s=100, marker="x")
    all_x.append(pred_person_location[0])
    all_y.append(pred_person_location[1])

    arrow_length = 0.5  # Scale for arrows
    
    ax3.arrow(pred_person_location[0], pred_person_location[1], arrow_length * predicted_heading[0], 
              arrow_length * predicted_heading[1], head_width=0.2, head_length=0.3, fc="orange", ec="orange", label="Shoulder Heading")
    ax3.arrow(robot_location[0], robot_location[1], 0.5 * robot_heading[0], 
              0.5 * robot_heading[1], head_width=0.2, head_length=0.3, fc="green", ec="green", label="GT Robot Heading")
    x_min, x_max = min(all_x) - 2, max(all_x) + 2
    y_min, y_max = min(all_y) - 2, max(all_y) + 2
    ax3.set_xlim(x_min, x_max)
    ax3.set_ylim(y_min, y_max)
    ax3.set_xlabel("X (meters)")
    ax3.set_ylabel("Y (meters)")
    ax3.set_title("Top-Down View")
    ax3.legend()
    ax3.grid()

    # New Camera FOV Orientation plot
    ax4 = fig.add_subplot(1, 4, 4)  # Add a fourth subplot
    ax4.scatter(0, 0, color="black", label="Camera", s=100, marker="o")  # Camera at origin
    ax4.scatter(pred_human_relative_loc[0], mean_depth, color="red", label="Predicted Person", s=100, marker="x")  # Person

    # Calculate and plot original skeleton heading
    left_shoulder_idx = keypoint_index['left_shoulder']
    right_shoulder_idx = keypoint_index['right_shoulder']
    hip_idx = keypoint_index['root']

    # Use the unrotated skeleton (first_person_3d) for the original heading
    left_shoulder = first_person_3d[left_shoulder_idx][:3]
    right_shoulder = first_person_3d[right_shoulder_idx][:3]
    hip = first_person_3d[hip_idx][:3]

    original_heading = -1 * np.cross(right_shoulder - hip, left_shoulder - hip)
    original_heading = original_heading[:2] / np.linalg.norm(original_heading[:2])  # Normalize in 2D plane

    # Plot original skeleton heading as an arrow from predicted person location
    arrow_length = 0.5  # Scale the arrow length for visibility
    ax4.arrow(pred_human_relative_loc[0], mean_depth, arrow_length * original_heading[0], arrow_length * original_heading[1],
              head_width=0.1, head_length=0.2, fc="orange", ec="orange", label="Original Skeleton Heading")

    # Plot styling
    ax4.set_xlim(-2, 2)  # Adjust limits for better visibility
    ax4.set_ylim(0, 5)   # Adjust depth range
    ax4.set_xlabel("X (meters)")
    ax4.set_ylabel("Y (meters)")
    ax4.set_title("Camera FOV Orientation")
    ax4.legend()
    ax4.grid()

    # Save the updated figure
    plt.tight_layout()
    print("Saving to", save_path)
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
