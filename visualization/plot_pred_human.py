# plot_pred_human.py: plots several mental models scatter plots as well as the robot's RGB, segmentation, and depth images

import matplotlib.patches, matplotlib.text, matplotlib.lines, matplotlib.collections
import matplotlib.pyplot as plt
import numpy as np
import cv2
import math

class PlotPredHuman():
    def __init__(self, classes, class_to_class_id, class_id_to_color_map, use_gt_semantics):
        self.classes = classes
        self.class_to_class_id = class_to_class_id
        self.class_id_to_color_map = class_id_to_color_map
        self.use_gt_semantics = use_gt_semantics

        self.map_boundaries_path = "./map_boundaries.png"
        self.plot_scatter_center = [-1,-4]
        self.plot_scatter_dim = 23
        self.map_boundaries = cv2.imread(self.map_boundaries_path, cv2.IMREAD_GRAYSCALE)
        self.map_image_dims = (self.map_boundaries.shape[1], self.map_boundaries.shape[0])
        self.map_boundaries = np.dstack((self.map_boundaries, self.map_boundaries, self.map_boundaries))

        self.fig = plt.figure(figsize=(12,6))
        self.fig.tight_layout()
        plt.subplots_adjust(hspace=.6)

        # scatter plot for robot mm
        self.ax_scatter_robot, self.plot_scatter_robot = self.__create_scatter_axis__(244, "Robot Semantic Map", xlim=((self.plot_scatter_center[0] - self.plot_scatter_dim / 2, self.plot_scatter_center[0] + self.plot_scatter_dim / 2)), ylim=(self.plot_scatter_center[1] - self.plot_scatter_dim / 2, self.plot_scatter_center[1] + self.plot_scatter_dim / 2), add_map=True)

        # add shading
        box_alpha = 0.2
        box_width = 0.17
        box_height = 0.88
        # sensors
        self.ax_scatter_robot.add_patch(matplotlib.patches.FancyBboxPatch([0.123, (1 - box_height) / 2], box_width, box_height, boxstyle="Round, pad=0, rounding_size=0.01", linewidth=0, ec='r', fc='r', alpha=box_alpha, clip_on=False, transform=self.fig.transFigure, zorder=-10))
        # detection
        self.ax_scatter_robot.add_patch(matplotlib.patches.FancyBboxPatch([0.326, (1 - box_height) / 2], box_width, box_height, boxstyle="Round, pad=0, rounding_size=0.01", linewidth=0, ec='orange', fc='orange', alpha=box_alpha, clip_on=False, transform=self.fig.transFigure, zorder=-10))
        # belief states
        self.ax_scatter_robot.add_patch(matplotlib.patches.FancyBboxPatch([0.731, (1 - box_height) / 2], box_width, box_height, boxstyle="Round, pad=0, rounding_size=0.01", linewidth=0, ec='blue', fc='blue', alpha=box_alpha, clip_on=False, transform=self.fig.transFigure, zorder=-10))
        # projection
        minibox_height = 0.403
        self.ax_scatter_robot.add_patch(matplotlib.patches.FancyBboxPatch([0.528, (1 - box_height) / 2 + 0.476], box_width, minibox_height, boxstyle="Round, pad=0, rounding_size=0.01", linewidth=0, ec='purple', fc='purple', alpha=box_alpha, clip_on=False, transform=self.fig.transFigure, zorder=-10))
        # gaze region
        self.ax_scatter_robot.add_patch(matplotlib.patches.FancyBboxPatch([0.528, (1 - box_height) / 2], box_width, minibox_height, boxstyle="Round, pad=0, rounding_size=0.01", linewidth=0, ec='purple', fc='purple', alpha=box_alpha, clip_on=False, transform=self.fig.transFigure, zorder=-10))

        # scatter plot for predicted human mm
        self.ax_scatter_human, self.plot_scatter_human = self.__create_scatter_axis__(248, "Inferred Human Map", xlim=((self.plot_scatter_center[0] - self.plot_scatter_dim / 2, self.plot_scatter_center[0] + self.plot_scatter_dim / 2)), ylim=(self.plot_scatter_center[1] - self.plot_scatter_dim / 2, self.plot_scatter_center[1] + self.plot_scatter_dim / 2), add_map=True)

        # robot RGB view axis
        self.ax_rgb_robot = self.fig.add_subplot(241)
        self.ax_rgb_robot.set_xlim(0, 512)
        self.ax_rgb_robot.set_ylim(0, 512)
        self.ax_rgb_robot.set_aspect('equal')
        self.ax_rgb_robot.set_axis_off()
        self.ax_rgb_robot.set_title("Robot's RGB Camera")
        self.plot_rgb_robot = self.ax_rgb_robot.imshow(np.zeros((512, 512, 3)))

        # robot RGB view axis
        self.ax_obj_robot = self.fig.add_subplot(242)
        self.ax_obj_robot.set_xlim(0, 512)
        self.ax_obj_robot.set_ylim(0, 512)
        self.ax_obj_robot.set_aspect('equal')
        self.ax_obj_robot.set_axis_off()
        self.ax_obj_robot.set_title("Object Detection")
        self.plot_obj_robot = self.ax_obj_robot.imshow(np.zeros((512, 512, 3)))

        # scatter plot for robot detected objects (local)
        self.ax_detections_local, self.plot_detections_local = self.__create_scatter_axis__(243, "Detections (Local)", xlim=(-5, 5), ylim=(0, 10))
        # draw a black circle of radius 10 centered at 0,0
        self.ax_detections_local.add_patch(matplotlib.patches.Circle((0, 0), 5, edgecolor="grey", facecolor="none", linewidth=1, linestyle="dashed"))
        self.ax_detections_local.add_patch(matplotlib.patches.Circle((0, 0), 10, edgecolor="lightgrey", facecolor="none", linewidth=1, linestyle="dashed"))
        self.ax_detections_local.add_patch(matplotlib.patches.Circle((0, 0), 15, edgecolor="gainsboro", facecolor="none", linewidth=1, linestyle="dashed"))
        # draw FOV lines
        fov, dist = math.radians(90 - 40 / 2), 20
        self.ax_detections_local.add_line(matplotlib.lines.Line2D([0, 0], [0, 20], color="gainsboro", linewidth=1, linestyle="dashed"))
        self.ax_detections_local.add_line(matplotlib.lines.Line2D([0, dist*math.cos(fov)], [0, dist*math.sin(fov)], color="gainsboro", linewidth=1, linestyle="dashed"))
        self.ax_detections_local.add_line(matplotlib.lines.Line2D([0, -1 * dist*math.cos(fov)], [0, dist*math.sin(fov)], color="gainsboro", linewidth=1, linestyle="dashed"))

        # trajectory view axis
        self.ax_human_trajectory, self.plot_human_trajectory = self.__create_scatter_axis__(247, "Inferred Gaze Region", xlim=(0, 512), ylim=(0, 512), add_map=True)
        self.plot_human_trajectory = self.ax_human_trajectory.imshow(np.ones((512, 512, 3)))

        # human RGB view axis
        self.ax_depth_robot = self.fig.add_subplot(245)
        self.ax_depth_robot.set_xlim(0, 512)
        self.ax_depth_robot.set_ylim(0, 512)
        self.ax_depth_robot.set_aspect('equal')
        self.ax_depth_robot.set_axis_off()
        self.ax_depth_robot.set_title("Robot's Depth Camera")
        self.plot_depth_robot = self.ax_depth_robot.imshow(np.zeros((512, 512, 3)))

        # pose detection axis
        self.ax_pose_detect = self.fig.add_subplot(246)
        self.ax_pose_detect.set_xlim(0, 512)
        self.ax_pose_detect.set_ylim(0, 512)
        self.ax_pose_detect.set_aspect('equal')
        self.ax_pose_detect.set_axis_off()
        self.ax_pose_detect.set_title("Pose Detection")
        self.plot_pose_detect = self.ax_pose_detect.imshow(np.ones((512, 512, 3)) * 255)

        # pose lines
        self.pose_lines = []

        # annotate the figure
        # self.fig.text(0.01, 0.99, "", ha='left', va='top', fontsize=14, color='black')
        # classes_vert = 0.90
        # self.fig.text(0.01, classes_vert, "Classes:", ha='left', va='top', fontsize=10, color='black')
        # for i, c in enumerate(classes):
        #     classes_vert -= 0.018
        #     self.fig.text(0.01, classes_vert, "    ҉  " + c, ha='left', va='top', fontsize=5, color=class_id_to_color_map[i])
        # self.text_frame = self.fig.text(0.99, 0.01, "Frame: N/A", ha="right", va="bottom", fontsize=10, color="black")

        # lookups
        self.keypoint_index = {
            'root': 0, 'right_hip': 1, 'right_knee': 2, 'right_foot': 3,
            'left_hip': 4, 'left_knee': 5, 'left_foot': 6, 'spine': 7,
            'thorax': 8, 'neck_base': 9, 'head': 10, 'left_shoulder': 11,
            'left_elbow': 12, 'left_wrist': 13, 'right_shoulder': 14,
            'right_elbow': 15, 'right_wrist': 16
        }

        self.skeleton_links = [
            ('root', 'left_hip'), ('left_hip', 'left_knee'), ('left_knee', 'left_foot'),
            ('root', 'right_hip'), ('right_hip', 'right_knee'), ('right_knee', 'right_foot'),
            ('root', 'spine'), ('spine', 'thorax'), ('thorax', 'neck_base'),
            ('neck_base', 'head'), ('thorax', 'left_shoulder'), ('left_shoulder', 'left_elbow'),
            ('left_elbow', 'left_wrist'), ('thorax', 'right_shoulder'), ('right_shoulder', 'right_elbow'),
            ('right_elbow', 'right_wrist')
        ]

    def update(self, robot_mm, pred_human_mm, gt_human_mm, agent_pose, gt_human_pose, robot_detected_objects, human_detections, human_trajectory_view, objects_visible_to_human, rgb_robot, depth_robot, rgb_human, frame_num, text_labels=False):
        # initialize the mental model scatter points
        mm_points_x_robot, mm_points_y_robot, mm_points_z_robot = [], [], []
        mm_points_x_human, mm_points_y_human, mm_points_z_human = [], [], []
        mm_points_x_human_true, mm_points_y_human_true, mm_points_z_human_true = [], [], []

        rgb_robot_raw = rgb_robot.copy()

        # add the robot agent
        mm_points_x_robot.append(agent_pose[0][0])  # east
        mm_points_y_robot.append(agent_pose[0][1])  # north
        mm_points_z_robot.append(agent_pose[0][2])  # vertical
        plot_colors_robot, plot_colors_human, plot_colors_human_true = [(0, 0, 0, 1)], [], []  # plot colors of mm points, initialize the robot's with the agent pose

        # fade out the previous motion colors
        for i, line in enumerate(self.pose_lines):
            alpha = line.get_alpha()
            if alpha is not None:  # if the line has an alpha value
                if alpha <= 0.05:  # delete if going invisible
                    del self.pose_lines[i]
                    continue
                line.set_alpha(alpha - 0.1)  # decrease alpha

        for o in robot_mm.dsg.objects:  # robot mental model objects
            mm_points_x_robot.append(robot_mm.dsg.objects[o]["x"])
            mm_points_y_robot.append(robot_mm.dsg.objects[o]["y"])
            plot_colors_robot.append(self.class_id_to_color_map[self.class_to_class_id[robot_mm.dsg.objects[o]["class"]]])

        for o in pred_human_mm.dsg.objects:  # predicted human mental model objects
            mm_points_x_human.append(pred_human_mm.dsg.objects[o]["x"])
            mm_points_y_human.append(pred_human_mm.dsg.objects[o]["y"])
            plot_colors_human.append(self.class_id_to_color_map[self.class_to_class_id[pred_human_mm.dsg.objects[o]["class"]]])

        for o in gt_human_mm.dsg.objects:  # true human mental model objects
            mm_points_x_human_true.append(gt_human_mm.dsg.objects[o]["x"])
            mm_points_y_human_true.append(gt_human_mm.dsg.objects[o]["y"])
            plot_colors_human_true.append(self.class_id_to_color_map[self.class_to_class_id[gt_human_mm.dsg.objects[o]["class"]]])

        # robot pose
        line = self.ax_scatter_robot.axes.plot((agent_pose[0][0], agent_pose[0][0] + agent_pose[-1][0]), (agent_pose[0][1], agent_pose[0][1] + agent_pose[-1][1]), color=(0, 0, 0), alpha=1.0)
        self.pose_lines.append(line[0])
        mm_points_x_robot.append(self.convert_continuous_to_plot(agent_pose[0][0]))
        mm_points_y_robot.append(self.convert_continuous_to_plot(agent_pose[0][1]))
        plot_colors_robot.append((0,0,0))

        # remove previous patches from robot RGB
        for patch in self.ax_pose_detect.get_children():
            if (patch.__class__ == matplotlib.patches.Rectangle and "type" in dir(patch)) or patch.__class__ == matplotlib.lines.Line2D or patch.__class__ == matplotlib.collections.PathCollection:
                patch.remove()

        # add human to belief state
        for detected_human in human_detections[0]:
            if "pose" not in detected_human:
                continue
            human_pose = detected_human["pose"]
            human_direction = detected_human["direction"]
            line = self.ax_scatter_robot.axes.plot((human_pose[0], human_pose[0] + human_direction[0]), (human_pose[1], human_pose[1] + human_direction[1]), color=(0, 0, 1), alpha=1.0)
            self.pose_lines.append(line[0])
            mm_points_x_robot.append(human_pose[0])
            mm_points_y_robot.append(human_pose[1])
            plot_colors_robot.append((0,0,1))
            human_angle = np.arctan2(human_direction[1], human_direction[0])
            # draw V lines for the human's field of view
            fov = pred_human_mm.fov
            line_length = 5
            line = self.ax_scatter_robot.axes.plot((human_pose[0], human_pose[0] + line_length * np.cos(human_angle - np.deg2rad(fov / 2))), (human_pose[1], human_pose[1] + line_length * np.sin(human_angle - np.deg2rad(fov / 2))), color=(0, 0, 1), alpha=0.2)
            self.pose_lines.append(line[0])
            line = self.ax_scatter_robot.axes.plot((human_pose[0], human_pose[0] + line_length * np.cos(human_angle + np.deg2rad(fov / 2))), (human_pose[1], human_pose[1] + line_length * np.sin(human_angle + np.deg2rad(fov / 2))), color=(0, 0, 1), alpha=0.2)
            self.pose_lines.append(line[0])

            # segment out the person on the image for visualization
            rgb_robot[detected_human["seg mask"]] = np.array([255, 255, 255])

            # draw keypoints on Robot RGB view
            print("Plotting detected human")
            for kp1, kp2 in self.skeleton_links:
                idx1, idx2 = self.keypoint_index[kp1], self.keypoint_index[kp2]
                x1, y1 = detected_human["keypoints"][idx1][:2]
                x2, y2 = detected_human["keypoints"][idx2][:2]
                self.ax_pose_detect.plot([x1, x2], [rgb_robot.shape[0] - y1, rgb_robot.shape[0] - y2], color='blue', linewidth=2)
            for kp in detected_human["keypoints"]:
                x, y = kp[:2]
                self.ax_pose_detect.scatter(x, rgb_robot.shape[0] - y, color='red', s=30)

        # initialize the detected objects scatter points
        xs_detections_local, ys_detections_local, plot_colors_detections = [], [], []

        # object detection
        for object in robot_detected_objects:
            rgb_robot[object["seg mask"]] = self.class_id_to_color_map[object["class id"]][:3] * 255
            plot_colors_detections.append(self.class_id_to_color_map[object["class id"]])
            xs_detections_local.append(object["x local"])
            ys_detections_local.append(object["y local"])

        self.plot_detections_local.set_offsets(np.c_[xs_detections_local, ys_detections_local])
        self.plot_detections_local.set_color(plot_colors_detections)

        self.plot_scatter_robot.set_offsets(np.c_[mm_points_x_robot, mm_points_y_robot])
        self.plot_scatter_robot.set_color(plot_colors_robot)

        self.plot_scatter_human.set_offsets(np.c_[mm_points_x_human, mm_points_y_human])
        self.plot_scatter_human.set_color(plot_colors_human)

        # remove previous patches (circles around visible objects)
        for patch in self.ax_scatter_human.get_children():
            if patch.__class__ == matplotlib.patches.Circle:
                patch.remove()

        # show circles around the visible objects
        for object in objects_visible_to_human:
            circle = matplotlib.patches.Circle((object["x"], object["y"]), .5, edgecolor='blue', facecolor='none')
            self.ax_scatter_human.add_patch(circle)

        # update the rgb image
        self.plot_rgb_robot.set_data(rgb_robot_raw[::-1,:,:])
        self.plot_depth_robot.set_data(depth_robot[::-1,:,:] / 3.5)
        self.plot_obj_robot.set_data(rgb_robot[::-1,:,:])

        # update the human trajectory view
        if human_trajectory_view is not None:
            # set alpha of the human trajectory view to full
            human_trajectory_view[:,:,3] = 255
            self.plot_human_trajectory.set_data(np.rot90(human_trajectory_view[::-1,:,:], k=-1))

        # self.text_frame.set_text(f"Frame: {frame_num}")

    def convert_continuous_to_plot(self, val):
        return
    
    def __create_scatter_axis__(self, location, title, xlim=None, ylim=None, add_map=False):
        ax = self.fig.add_subplot(location)#, projection='3d')  # technically can be 3D, using 2D for simplicity
        if xlim is not None:
            ax.set_xlim(xlim)
        else:
            ax.set_xlim((self.plot_scatter_center[0] - self.plot_scatter_dim / 2, self.plot_scatter_center[0] + self.plot_scatter_dim / 2))
        if ylim is not None:
            ax.set_ylim(ylim)
        else:
            ax.set_ylim((self.plot_scatter_center[1] - self.plot_scatter_dim / 2, self.plot_scatter_center[1] + self.plot_scatter_dim / 2))
        ax.set_aspect('equal')
        ax.set_axis_off()
        ax.set_title(title)
        plot = ax.scatter([], [], s=10)
        # draw a white rectangle covering the whole plot
        # get the boundaries of the plot
        min_x, max_x = ax.get_xlim()
        min_y, max_y = ax.get_ylim()
        ax.add_patch(matplotlib.patches.Rectangle((min_x, min_y), max_x - min_x, max_y - min_y, edgecolor="white", facecolor="white", zorder=-3))
        if add_map:
            # add map background
            ax.imshow(self.map_boundaries, extent=[self.plot_scatter_center[0] - self.plot_scatter_dim / 2, self.plot_scatter_center[0] + self.plot_scatter_dim / 2, self.plot_scatter_center[1] - self.plot_scatter_dim / 2, self.plot_scatter_center[1] + self.plot_scatter_dim / 2])
        return ax, plot
