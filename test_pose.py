from detection import detect
from pose_estimation import pose
import cv2, numpy, torch

# load example image
bgr = cv2.imread("episodes/episode_2024-09-04-16-32_agents_2_run_19/0/Action_0017_0_normal.png")
img = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

# get bounding boxes
objects, _ = detect.detect(img, ["human"])
boxes = torch.Tensor([[x for sublist in objects[i]["box"] for x in sublist] for i in range(len(objects))])

# init pose object
pose_detector = pose.PoseDetector(device="mps")

# run AlphaPose
keypoints = pose_detector.get_2d_pose(img, boxes)

# run MotionBERT
pose_3d = pose_detector.get_3d_pose(img, boxes, keypoints)

pose_image = pose_detector.draw_poses_2d(img, keypoints)
cv2.imwrite("pose_image.png", cv2.cvtColor(pose_image, cv2.COLOR_RGB2BGR))
print(objects)
