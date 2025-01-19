# plot_pred_human.py: plots several mental models scatter plots as well as the robot's RGB, segmentation, and depth images

import matplotlib.patches, matplotlib.text, matplotlib.lines, matplotlib.collections
import matplotlib.pyplot as plt
import numpy as np
import cv2
import math

class PlotPerception():
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

        self.fig = plt.figure(figsize=(10,6))
        self.fig.tight_layout()

        # scatter plot for robot mm
        self.ax_scatter_mm, self.plot_scatter_mm = self.__create_scatter_axis__(231, "Semantic Map")

        # add map background
        self.ax_scatter_robot_map = self.fig.add_axes(self.ax_scatter_mm.get_position(), frame_on=False, zorder=-1)
        self.ax_scatter_robot_map.imshow(self.map_boundaries)
        self.ax_scatter_robot_map.set_axis_off()

        # scatter plot for robot detected objects (global)
        self.ax_detections, self.plot_detections = self.__create_scatter_axis__(232, "Detections (Global)")

        # scatter plot for robot detected objects (local)
        self.ax_detections_local, self.plot_detections_local = self.__create_scatter_axis__(233, "Detections (Local)", xlim=(-5, 5), ylim=(0, 10))
        # draw a black circle of radius 10 centered at 0,0
        self.ax_detections_local.add_patch(matplotlib.patches.Circle((0, 0), 5, edgecolor="grey", facecolor="none", linewidth=1, linestyle="dashed"))
        self.ax_detections_local.add_patch(matplotlib.patches.Circle((0, 0), 10, edgecolor="lightgrey", facecolor="none", linewidth=1, linestyle="dashed"))
        self.ax_detections_local.add_patch(matplotlib.patches.Circle((0, 0), 15, edgecolor="gainsboro", facecolor="none", linewidth=1, linestyle="dashed"))
        # draw FOV lines
        fov, dist = math.radians(90 - 40 / 2), 20
        self.ax_detections_local.add_line(matplotlib.lines.Line2D([0, 0], [0, 20], color="gainsboro", linewidth=1, linestyle="dashed"))
        self.ax_detections_local.add_line(matplotlib.lines.Line2D([0, dist*math.cos(fov)], [0, dist*math.sin(fov)], color="gainsboro", linewidth=1, linestyle="dashed"))
        self.ax_detections_local.add_line(matplotlib.lines.Line2D([0, -1 * dist*math.cos(fov)], [0, dist*math.sin(fov)], color="gainsboro", linewidth=1, linestyle="dashed"))

        # robot RGB view axis
        self.ax_rgb, self.plot_rgb = self.__create_image_axis__(234, "RGB Camera")

        # robot depth view axis
        self.ax_depth, self.plot_depth = self.__create_image_axis__(235, "Depth Camera")

        # robot object annotation axis
        self.ax_annotations, self.plot_annotations = self.__create_image_axis__(236, "Object Annotations")

        # pose lines
        self.pose_lines = []

        # annotate the figure
        self.fig.text(0.01, 0.99, "Perception Stack Visualization", ha='left', va='top', fontsize=14, color='black')
        classes_vert = 0.90
        self.fig.text(0.01, classes_vert, "Classes:", ha='left', va='top', fontsize=10, color='black')
        for i, c in enumerate(classes):
            classes_vert -= 0.018
            self.fig.text(0.01, classes_vert, "    ҉  " + c, ha='left', va='top', fontsize=5.5, color=class_id_to_color_map[i])
        self.text_frame = self.fig.text(0.99, 0.01, "Frame: N/A", ha="right", va="bottom", fontsize=10, color="black")

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

    def update(self, robot_mm, agent_pose, robot_detected_objects, human_detections, rgb_robot, depth_robot, frame_num):
        """
        Update the visualization with the latest mental models and detections
        
        Args:
            robot_mm (mental_model.MentalModel): the robot's mental model
            agent_pose (tuple): the agent's pose, (hip location XY, direction vector)
            robot_detected_objects (list): the detected objects in the robot's view
            human_detections (list): the detected humans in the robot's view
            rgb_robot (np.array): the robot's RGB image
            depth_robot (np.array): the robot's depth image
            frame_num (int): the frame number
        """
        # initialize the mental model scatter points
        mm_points_x_robot, mm_points_y_robot, mm_points_z_robot = [], [], []

        # add the robot agent
        mm_points_x_robot.append(agent_pose[0][0])  # east
        mm_points_y_robot.append(agent_pose[0][1])  # north
        mm_points_z_robot.append(agent_pose[0][2])  # vertical
        plot_colors_robot = [(0, 0, 0, 1)]  # plot colors of mm points, initialize the robot's with the agent pose

        # fade out the previous motion colors
        for i, line in enumerate(self.pose_lines):
            alpha = line.get_alpha()
            if alpha is not None:  # if the line has an alpha value
                if alpha <= 0.05:  # delete if going invisible
                    del self.pose_lines[i]
                    continue
                line.set_alpha(alpha - 0.1)  # decrease alpha

        for o in robot_mm.dsg.objects:  # robot mental model objects
            if robot_mm.dsg.objects[o]["class"] not in self.class_to_class_id:
                continue
            mm_points_x_robot.append(robot_mm.dsg.objects[o]["x"])
            mm_points_y_robot.append(robot_mm.dsg.objects[o]["y"])
            plot_colors_robot.append(self.class_id_to_color_map[self.class_to_class_id[robot_mm.dsg.objects[o]["class"]]])

        # robot pose
        line = self.ax_scatter_mm.axes.plot((agent_pose[0][0], agent_pose[0][0] + agent_pose[-1][0]), (agent_pose[0][1], agent_pose[0][1] + agent_pose[-1][1]), color=(0, 0, 0), alpha=1.0)
        self.pose_lines.append(line[0])
        mm_points_x_robot.append(self.convert_continuous_to_plot(agent_pose[0][0]))
        mm_points_y_robot.append(self.convert_continuous_to_plot(agent_pose[0][1]))
        plot_colors_robot.append((0,0,0))

        # remove previous patches from robot RGB
        for patch in self.ax_rgb.get_children():
            if (patch.__class__ == matplotlib.patches.Rectangle and "type" in dir(patch)) or patch.__class__ == matplotlib.lines.Line2D or patch.__class__ == matplotlib.collections.PathCollection:
                patch.remove()

        # human pose
        for detected_human in human_detections[0]:
            if "pose" not in detected_human:
                continue
            human_pose = detected_human["pose"]
            human_direction = detected_human["direction"]
            line = self.ax_scatter_mm.axes.plot((human_pose[0], human_pose[0] + human_direction[0]), (human_pose[1], human_pose[1] + human_direction[1]), color=(0, 0, 1), alpha=1.0)
            self.pose_lines.append(line[0])
            mm_points_x_robot.append(human_pose[0])
            mm_points_y_robot.append(human_pose[1])
            plot_colors_robot.append((0,0,1))
            human_angle = np.arctan2(human_direction[1], human_direction[0])

            # draw V lines for the human's field of view
            fov = 40
            line_length = 5
            line = self.ax_scatter_mm.axes.plot((human_pose[0], human_pose[0] + line_length * np.cos(human_angle - np.deg2rad(fov / 2))), (human_pose[1], human_pose[1] + line_length * np.sin(human_angle - np.deg2rad(fov / 2))), color=(0, 0, 1), alpha=0.2)
            self.pose_lines.append(line[0])
            line = self.ax_scatter_mm.axes.plot((human_pose[0], human_pose[0] + line_length * np.cos(human_angle + np.deg2rad(fov / 2))), (human_pose[1], human_pose[1] + line_length * np.sin(human_angle + np.deg2rad(fov / 2))), color=(0, 0, 1), alpha=0.2)
            self.pose_lines.append(line[0])

            # segment out the person on the image for visualization
            rgb_robot[detected_human["seg mask"]] = np.array([255, 255, 255])

            # draw a rectangle bounding box
            bbox = [y for x in detected_human["box"] for y in x]
            rect = matplotlib.patches.Rectangle((bbox[0], rgb_robot.shape[1] - bbox[1]), bbox[2] - bbox[0], bbox[1] - bbox[3], linewidth=2, edgecolor='r', facecolor='none')
            rect.type = "bounding box"
            self.ax_rgb.add_patch(rect)

            # draw keypoints on Robot RGB view
            print("Plotting detected human")
            for kp1, kp2 in self.skeleton_links:
                idx1, idx2 = self.keypoint_index[kp1], self.keypoint_index[kp2]
                x1, y1 = detected_human["keypoints"][idx1][:2]
                x2, y2 = detected_human["keypoints"][idx2][:2]
                self.ax_rgb.plot([x1, x2], [rgb_robot.shape[0] - y1, rgb_robot.shape[0] - y2], color='blue', linewidth=2)
            for kp in detected_human["keypoints"]:
                x, y = kp[:2]
                self.ax_rgb.scatter(x, rgb_robot.shape[0] - y, color='red', s=30)

        self.plot_scatter_mm.set_offsets(np.c_[mm_points_x_robot, mm_points_y_robot])
        self.plot_scatter_mm.set_color(plot_colors_robot)

        # remove text annotations from the annotations axis
        for text in self.ax_annotations.texts:
            text.remove()

        # initialize the detected objects scatter points
        xs_detections, ys_detections, plot_colors_detections = [], [], []
        xs_detections_local, ys_detections_local = [], []

        # object detection
        for object in robot_detected_objects:
            # print all unique values of seg mask
            rgb_robot[object["seg mask"]] = self.class_id_to_color_map[object["class id"]][:3] * 255
            xs_detections.append(object["x"])
            ys_detections.append(object["y"])
            plot_colors_detections.append(self.class_id_to_color_map[object["class id"]])

            xs_detections_local.append(object["x local"])
            ys_detections_local.append(object["y local"])

            # add a text annotation on the annotations axis
            # get the average pixel coordinate where the seg mask is True
            y, x = np.mean(np.where(object["seg mask"] == 1), axis=1)
            self.ax_annotations.text(x, object["seg mask"].shape[1] - y, object["class"], fontsize=8, color=self.class_id_to_color_map[object["class id"]])

        self.plot_detections.set_offsets(np.c_[xs_detections, ys_detections])
        self.plot_detections.set_color(plot_colors_detections)

        self.plot_detections_local.set_offsets(np.c_[xs_detections_local, ys_detections_local])
        self.plot_detections_local.set_color(plot_colors_detections)

        # remove line patches from plot_detections
        for patch in self.ax_detections.get_children():
            if patch.__class__ == matplotlib.lines.Line2D:
                patch.remove()

        # draw FOV lines
        fov, dist = math.radians(40 / 2), 20
        # get the angle from the agent's direction vector
        angle = math.atan2(agent_pose[1][1], agent_pose[1][0])

        self.ax_detections.add_line(matplotlib.lines.Line2D([agent_pose[0][0], agent_pose[0][0]+dist*math.cos(angle)], [agent_pose[0][1], agent_pose[0][1]+dist*math.sin(angle)], color="gainsboro", linewidth=1, linestyle="dashed"))
        self.ax_detections.add_line(matplotlib.lines.Line2D([agent_pose[0][0], agent_pose[0][0]+dist*math.cos(angle+fov)], [agent_pose[0][1], agent_pose[0][1]+dist*math.sin(angle+fov)], color="gainsboro", linewidth=1, linestyle="dashed"))
        self.ax_detections.add_line(matplotlib.lines.Line2D([agent_pose[0][0], agent_pose[0][0]+dist*math.cos(angle-fov)], [agent_pose[0][1], agent_pose[0][1]+dist*math.sin(angle-fov)], color="gainsboro", linewidth=1, linestyle="dashed"))

        # update the rgb image
        self.plot_rgb.set_data(rgb_robot[::-1,:,:])
        self.text_frame.set_text(f"Frame: {frame_num}")

        # update the depth image
        self.plot_depth.set_data(depth_robot[::-1,:,:] / 5)

    def convert_continuous_to_plot(self, val):
        return

    def __create_scatter_axis__(self, location, title, xlim=None, ylim=None):
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
        return ax, plot
    
    def __create_image_axis__(self, location, title):
        ax = self.fig.add_subplot(location)
        ax.set_xlim(0, 512)
        ax.set_ylim(0, 512)
        ax.set_aspect('equal')
        ax.set_axis_off()
        ax.set_title(title)
        plot = ax.imshow(np.ones((512, 512, 3)) * 255)
        return ax, plot