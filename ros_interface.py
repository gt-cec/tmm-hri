import rospy
import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"
import cv2
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, Imu, CameraInfo
from nav_msgs.msg import Odometry
import geometry_msgs
import tf2_ros
from tf.transformations import quaternion_from_matrix, quaternion_from_euler, quaternion_about_axis
import numpy as np
import hrnet.hrnet

# initialize the node
rospy.init_node("smm_tri_ros_interface")
image_pub = rospy.Publisher('/tesse/left_cam/rgb/image_raw', Image, queue_size=100)
seg_pub = rospy.Publisher('/tesse/seg_cam/rgb/image_raw', Image, queue_size=100)
seg_info_pub = rospy.Publisher('/tesse/seg_cam/camera_info', CameraInfo, queue_size=100)
depth_pub = rospy.Publisher('/tesse/depth_cam/mono/image_raw', Image, queue_size=100)
# pose_pub = rospy.Publisher('/tesse/odom', Odometry, queue_size=100)
#pose_test_pub = rospy.Publisher('/tf', Odometry, queue_size=100)

# read the RGB/Depth/pose data for each frame and send it over ROS
def relay_output(folder="./Output", char=0):
    if folder[-1] == "/":
        folder = folder[:-1]

    # get the poses
    print("Reading poses")
    poses = {}
    with open("{}/human/{}/pd_human.txt".format(folder, char)) as f:
        for line in f.readlines()[1:]:
            vals = line.split(" ")
            frame = int(vals[0])
            # hip_x = float(vals[1])  # right
            # hip_y = float(vals[2])  # up
            # hip_z = float(vals[3])  # front
            poses[frame] = [float(x) for x in vals[1:]] #[hip_x, hip_y, hip_z]
    print("Done reading poses")
    
    # relay the images and poses
    bridge = CvBridge()
    rate = rospy.Rate(10)  # 10 fps
    for frame in poses:
        # read the RGB image
        image = cv2.imread("{}/human/{}/Action_{}_{}_normal.png".format(folder, char, "0"*(4-len(str(frame))) + str(frame), char))
        # read the depth image
        depth = cv2.imread("{}/human/{}/Action_{}_{}_depth.exr".format(folder, char, "0"*(4-len(str(frame))) + str(frame), char), cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH) * .2

        if depth.ndim == 3 and depth.shape[2] == 3:
            single_channel_image = depth[:, :, 0]  # Use the first channel
        else:
            single_channel_image = depth

        # Ensure the image is 32-bit float
        if single_channel_image.dtype != np.float32:
            single_channel_image = single_channel_image.astype(np.float32)
        
        depth = single_channel_image

        # cv2.imshow("rgb", image)
        # cv2.imshow("depth", depth)
        # cv2.waitKey(1)

        # convert to a ROS image        
        ros_image = bridge.cv2_to_imgmsg(image, encoding="bgr8")
        ros_image.header.frame_id = "left_cam"
        ros_depth = bridge.cv2_to_imgmsg(depth, encoding="32FC1")
        ros_depth.header.frame_id = "left_cam"
        ros_pose = Odometry()
        ros_pose.header.stamp = rospy.Time.now()
        ros_pose.header.frame_id = "world"  # Set the frame id (e.g., 'odom' or 'map')

        ros_test_pose = Odometry()
        ros_test_pose.header.stamp = rospy.Time.now()
        ros_test_pose.header.frame_id = "world"  # Set the frame id (e.g., 'odom' or 'map')
        
        # get locations of key bones
        hip_loc = extract_pose_loc_for_index(poses[frame], 0, cast_to_numpy_array=True)
        left_shoulder_loc = extract_pose_loc_for_index(poses[frame], 11, cast_to_numpy_array=True)
        right_shoulder_loc = extract_pose_loc_for_index(poses[frame], 12, cast_to_numpy_array=True)
        
        # calculate the orientation: plane from shoulders left/right and head, head should be slightly forward
        forward = np.cross(left_shoulder_loc - hip_loc, right_shoulder_loc - hip_loc)
        forward /= np.linalg.norm(forward)
        q = direction_to_quaternion(forward)
        
        ros_test_pose.pose.pose = geometry_msgs.msg.Pose(geometry_msgs.msg.Point(0, 0, 0), geometry_msgs.msg.Quaternion(*q))
        ros_pose.pose.pose = geometry_msgs.msg.Pose(geometry_msgs.msg.Point(hip_loc[0], hip_loc[1], hip_loc[2]), geometry_msgs.msg.Quaternion(*q))

        # get the segmentation map
        seg_img = hrnet.hrnet.segment(image)
        ros_seg = bridge.cv2_to_imgmsg(seg_img, encoding="bgr8")
        ros_seg.header.frame_id = "left_cam"
        ros_seg_info = CameraInfo(height=image.shape[0], width=image.shape[1])
        ros_seg_info.header.frame_id = "left_cam"

        # world transform
        # world_br = tf2_ros.TransformBroadcaster()
        # world_t = geometry_msgs.msg.TransformStamped()
        # world_t.header.stamp = rospy.Time.now()
        # world_t.header.frame_id = "world"
        # world_t.child_frame_id = "base_link_gt"
        # world_t.transform.translation.x = 0
        # world_t.transform.translation.y = 0
        # world_t.transform.translation.z = 0
        # world_t.transform.rotation.x = 0
        # world_t.transform.rotation.y = 0
        # world_t.transform.rotation.z = 0
        # world_t.transform.rotation.w = 1

        # base link transform
        base_link_br = tf2_ros.TransformBroadcaster()
        base_link_t = geometry_msgs.msg.TransformStamped()
        base_link_t.header.stamp = rospy.Time.now()
        base_link_t.header.frame_id = "base_link_gt"
        base_link_t.child_frame_id = "left_cam"
        base_link_t.transform.translation.x = ros_pose.pose.pose.position.x
        base_link_t.transform.translation.y = ros_pose.pose.pose.position.y
        base_link_t.transform.translation.z = ros_pose.pose.pose.position.z
        base_link_t.transform.rotation.x = q[0]
        base_link_t.transform.rotation.y = q[1]
        base_link_t.transform.rotation.z = q[2]
        base_link_t.transform.rotation.w = q[3]

        # left cam transform
        # left_cam_br = tf2_ros.TransformBroadcaster()
        # left_cam_t = geometry_msgs.msg.TransformStamped()
        # left_cam_t.header.stamp = rospy.Time.now()
        # left_cam_t.header.frame_id = "left_cam"
        # # left_cam_t.child_frame_id = 
        # left_cam_t.transform.translation.x = 0
        # left_cam_t.transform.translation.y = 0
        # left_cam_t.transform.translation.z = 0
        # left_cam_t.transform.rotation.x = 0
        # left_cam_t.transform.rotation.y = 0
        # left_cam_t.transform.rotation.z = 0
        # left_cam_t.transform.rotation.w = 0

        print("   publishing", frame)

        # publish it
        if not rospy.is_shutdown():
            image_pub.publish(ros_image)
            seg_pub.publish(ros_seg)
            seg_info_pub.publish(ros_seg_info)
            depth_pub.publish(ros_depth)
            # pose_pub.publish(ros_pose)  # not needed by the uhumans, I think because it's included in the /tf 
            # pose_test_pub.publish(ros_test_pose)
            base_link_br.sendTransform(base_link_t)
            # left_cam_br.sendTransform(left_cam_t)
            rate.sleep()
        else:
            break

# calculate the quaternion given a forward direction vector
def direction_to_quaternion(direction_vector):
    # normalize the vector
    direction_vector = np.array(direction_vector)  
    direction_vector /= np.linalg.norm(direction_vector)

    target_vector = np.array([-1.0, 0.0, 0.0])  # use the direction vector along the x axis
    
    # get the cross product and the angle
    axis = np.cross(target_vector, direction_vector)
    axis /= np.linalg.norm(axis)
    angle = np.arccos(np.dot(target_vector, direction_vector))
    
    # return the quaternion
    return quaternion_about_axis(angle, axis)

# get the x/y/z coordinates of an index from the pose array
def extract_pose_loc_for_index(pose_list, index, cast_to_numpy_array=False):
    r = [pose_list[3 * index + 0], pose_list[3 * index + 1], pose_list[3 * index + 2]]
    return np.array(r) if cast_to_numpy_array else r

if __name__ == "__main__":
    # initialize hrnet
    hrnet.hrnet.load_model()

    # run forever in a loop
    while not rospy.is_shutdown():
        relay_output()