# plot_pred_human.py: plots several mental models scatter plots as well as the robot's RGB, segmentation, and depth images

import matplotlib.patches, matplotlib.text
import matplotlib.pyplot as plt
import numpy as np
import cv2

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

        self.fig = plt.figure(figsize=(10,6))
        self.fig.tight_layout()

        # scatter plot for robot mm
        self.ax_scatter_robot = self.fig.add_subplot(231)#, projection='3d')  # technically can be 3D, using 2D for simplicity
        self.ax_scatter_robot.set_xlim((self.plot_scatter_center[0] - self.plot_scatter_dim / 2, self.plot_scatter_center[0] + self.plot_scatter_dim / 2))
        self.ax_scatter_robot.set_ylim((self.plot_scatter_center[1] - self.plot_scatter_dim / 2, self.plot_scatter_center[1] + self.plot_scatter_dim / 2))
        self.ax_scatter_robot.set_aspect('equal')
        self.ax_scatter_robot.set_axis_off()
        self.ax_scatter_robot.set_title("Robot Semantic Map")
        xs, ys = [], []
        self.plot_scatter_robot = self.ax_scatter_robot.scatter(xs, ys, s=10)

        # add map background
        self.ax_scatter_robot_map = self.fig.add_axes(self.ax_scatter_robot.get_position(), frame_on=False, zorder=-1)
        self.ax_scatter_robot_map.imshow(self.map_boundaries)
        self.ax_scatter_robot.set_aspect('equal')
        self.ax_scatter_robot_map.set_axis_off()

        # scatter plot for predicted human mm
        self.ax_scatter_human = self.fig.add_subplot(232)#, projection='3d')  # technically can be 3D, using 2D for simplicity
        self.ax_scatter_human.set_xlim((self.plot_scatter_center[0] - self.plot_scatter_dim / 2, self.plot_scatter_center[0] + self.plot_scatter_dim / 2))
        self.ax_scatter_human.set_ylim((self.plot_scatter_center[1] - self.plot_scatter_dim / 2, self.plot_scatter_center[1] + self.plot_scatter_dim / 2))
        self.ax_scatter_human.set_aspect('equal')
        self.ax_scatter_human.set_axis_off()
        self.ax_scatter_human.set_title("Predicted Human Semantic Map")
        xs, ys = [], []
        self.plot_scatter_human = self.ax_scatter_human.scatter(xs, ys, s=10)

        # add map background
        self.ax_scatter_human_map = self.fig.add_axes(self.ax_scatter_human.get_position(), frame_on=False, zorder=-1)
        self.ax_scatter_human_map.imshow(self.map_boundaries)
        self.ax_scatter_human_map.set_aspect('equal')
        self.ax_scatter_human_map.set_axis_off()

        # scatter plot for true human mm
        self.ax_scatter_human_true = self.fig.add_subplot(233)#, projection='3d')  # technically can be 3D, using 2D for simplicity
        self.ax_scatter_human_true.set_xlim((self.plot_scatter_center[0] - self.plot_scatter_dim / 2, self.plot_scatter_center[0] + self.plot_scatter_dim / 2))
        self.ax_scatter_human_true.set_ylim((self.plot_scatter_center[1] - self.plot_scatter_dim / 2, self.plot_scatter_center[1] + self.plot_scatter_dim / 2))
        self.ax_scatter_human_true.set_aspect('equal')
        self.ax_scatter_human_true.set_axis_off()
        self.ax_scatter_human_true.set_title("True Human Semantic Map")
        xs, ys = [], []
        self.plot_scatter_human_true = self.ax_scatter_human_true.scatter(xs, ys, s=10)

        # add map background
        self.ax_scatter_human_true_map = self.fig.add_axes(self.ax_scatter_human_true.get_position(), frame_on=False, zorder=-1)
        self.ax_scatter_human_true_map.imshow(self.map_boundaries)
        self.ax_scatter_human_true_map.set_aspect('equal')
        self.ax_scatter_human_true_map.set_axis_off()

        # robot RGB view axis
        self.ax_rgb_robot = self.fig.add_subplot(234)
        self.ax_rgb_robot.set_xlim(0, 512)
        self.ax_rgb_robot.set_ylim(0, 512)
        self.ax_rgb_robot.set_aspect('equal')
        self.ax_rgb_robot.set_axis_off()
        self.ax_rgb_robot.set_title("Robot RGB Camera")
        self.plot_rgb_robot = self.ax_rgb_robot.imshow(np.zeros((512, 512, 3)))

        # trajectory view axis
        self.ax_human_trajectory = self.fig.add_subplot(235)
        self.ax_human_trajectory.set_xlim(0, 512)
        self.ax_human_trajectory.set_ylim(0, 512)
        self.ax_human_trajectory.set_aspect('equal')
        self.ax_human_trajectory.set_axis_off()
        self.ax_human_trajectory.set_title("Human Trajectory View")
        self.plot_human_trajectory = self.ax_human_trajectory.imshow(np.ones((512, 512, 3)))

        # human RGB view axis
        self.ax_rgb_human = self.fig.add_subplot(236)
        self.ax_rgb_human.set_xlim(0, 512)
        self.ax_rgb_human.set_ylim(0, 512)
        self.ax_rgb_human.set_aspect('equal')
        self.ax_rgb_human.set_axis_off()
        self.ax_rgb_human.set_title("True Human RGB Camera")
        self.plot_rgb_human = self.ax_rgb_human.imshow(np.zeros((512, 512, 3)))

        # pose lines
        self.pose_lines = []

        # annotate the figure
        self.fig.text(0.01, 0.99, "Demo of Semantic Map Construction", ha='left', va='top', fontsize=14, color='black')
        classes_vert = 0.90
        self.fig.text(0.01, classes_vert, "Classes:", ha='left', va='top', fontsize=10, color='black')
        for i, c in enumerate(classes):
            classes_vert -= 0.025
            self.fig.text(0.01, classes_vert, "    ҉  " + c, ha='left', va='top', fontsize=8, color=class_id_to_color_map[i])
        self.text_frame = self.fig.text(0.99, 0.01, "Frame: N/A", ha="right", va="bottom", fontsize=10, color="black")

    def update(self, robot_mm, pred_human_mm, gt_human_mm, agent_pose, gt_human_pose, robot_detected_objects, human_detections, human_trajectory_view, objects_visible_to_human, rgb_robot, rgb_human, frame_num, text_labels=False):
        # initialize the mental model scatter points
        mm_points_x_robot, mm_points_y_robot, mm_points_z_robot = [], [], []
        mm_points_x_human, mm_points_y_human, mm_points_z_human = [], [], []
        mm_points_x_human_true, mm_points_y_human_true, mm_points_z_human_true = [], [], []

        # add the robot agent
        mm_points_x_robot.append(0)#agent_pose[0][0])  # east
        mm_points_y_robot.append(0)#agent_pose[0][1])  # north
        mm_points_z_robot.append(0)#agent_pose[0][2])  # vertical
        plot_colors_robot, plot_colors_human, plot_colors_human_true = [(0, 0, 0, 1)], [], []  # plot colors of mm points, initialize the robot's with the agent pose

        # fade out the previous motion colors
        for i, line in enumerate(self.pose_lines):
            alpha = line.get_alpha()
            if alpha is not None:  # if the line has an alpha value
                if alpha <= 0.05:  # delete if going invisible
                    del self.pose_lines[i]
                    continue
                line.set_alpha(alpha - 0.1)  # decrease alpha

        # for o in robot_mm.dsg.objects:  # robot mental model objects
        #     mm_points_x_robot.append(robot_mm.dsg.objects[o]["x"])
        #     mm_points_y_robot.append(robot_mm.dsg.objects[o]["y"])
        #     plot_colors_robot.append(self.class_id_to_color_map[self.class_to_class_id[robot_mm.dsg.objects[o]["class"]]])

        # for o in pred_human_mm.dsg.objects:  # predicted human mental model objects
        #     mm_points_x_human.append(pred_human_mm.dsg.objects[o]["x"])
        #     mm_points_y_human.append(pred_human_mm.dsg.objects[o]["y"])
        #     plot_colors_human.append(self.class_id_to_color_map[self.class_to_class_id[pred_human_mm.dsg.objects[o]["class"]]])

        # for o in gt_human_mm.dsg.objects:  # true human mental model objects
        #     mm_points_x_human_true.append(gt_human_mm.dsg.objects[o]["x"])
        #     mm_points_y_human_true.append(gt_human_mm.dsg.objects[o]["y"])
        #     plot_colors_human_true.append(self.class_id_to_color_map[self.class_to_class_id[gt_human_mm.dsg.objects[o]["class"]]])

        # robot pose
        line = self.ax_scatter_robot.axes.plot((agent_pose[0][0], agent_pose[0][0] + agent_pose[-1][0]), (agent_pose[0][1], agent_pose[0][1] + agent_pose[-1][1]), color=(0, 0, 0), alpha=1.0)
        self.pose_lines.append(line[0])
        mm_points_x_robot.append(self.convert_continuous_to_plot(agent_pose[0][0]))
        mm_points_y_robot.append(self.convert_continuous_to_plot(agent_pose[0][1]))
        plot_colors_robot.append((0,0,0))

        # gt human pose
        line = self.ax_scatter_human_true.axes.plot((gt_human_pose[0][0], gt_human_pose[0][0] + gt_human_pose[-1][0]), (gt_human_pose[0][1], gt_human_pose[0][1] + gt_human_pose[-1][1]), color=(0, 0, 1), alpha=1.0)
        self.pose_lines.append(line[0])
        mm_points_x_human_true.append(gt_human_pose[0][0])
        mm_points_y_human_true.append(gt_human_pose[0][1])
        plot_colors_human_true.append((0,0,1))

        # human pose
        for detected_human in human_detections[0]:
            human_pose = detected_human["pose"]
            human_direction = detected_human["direction"]
            print("!!!", human_pose)
            line = self.ax_scatter_robot.axes.plot((human_pose[0], human_pose[0] + human_direction[0]), (human_pose[1], human_pose[1] + human_direction[1]), color=(0, 0, 1), alpha=1.0)
            self.pose_lines.append(line[0])
            # mm_points_x_robot.append(human_pose[0][0])
            # mm_points_y_robot.append(human_pose[0][1])
            # plot_colors_robot.append((0,0,1))
            human_angle = np.arctan2(human_direction[1], human_direction[0])
            # draw V lines for the human's field of view
            fov = pred_human_mm.fov
            line = self.ax_scatter_robot.axes.plot((human_pose[0], human_pose[0] + 5 * np.cos(human_angle - np.deg2rad(fov / 2))), (human_pose[1], human_pose[1] + 5 * np.sin(human_angle - np.deg2rad(fov / 2))), color=(0, 0, 1), alpha=0.2)
            self.pose_lines.append(line[0])
            line = self.ax_scatter_robot.axes.plot((human_pose[0], human_pose[0] + 5 * np.cos(human_angle + np.deg2rad(fov / 2))), (human_pose[1], human_pose[1] + 5 * np.sin(human_angle + np.deg2rad(fov / 2))), color=(0, 0, 1), alpha=0.2)
            self.pose_lines.append(line[0])

        self.plot_scatter_robot.set_offsets(np.c_[mm_points_x_robot, mm_points_y_robot])
        self.plot_scatter_robot.set_color(plot_colors_robot)

        # if mm_points_x_human != [] and mm_points_y_human != [] and mm_points_z_human != []:  # update the plot points and colors
        self.plot_scatter_human.set_offsets(np.c_[mm_points_x_human, mm_points_y_human])
        self.plot_scatter_human.set_color(plot_colors_human)

        # if mm_points_x_human_true != [] and mm_points_y_human_true != [] and mm_points_z_human_true != []:  # update the plot points and colors
        self.plot_scatter_human_true.set_offsets(np.c_[mm_points_x_human_true, mm_points_y_human_true])
        self.plot_scatter_human_true.set_color(plot_colors_human_true)

        # remove previous patches (circles around visible objects)
        for patch in self.ax_scatter_human.get_children():
            if patch.__class__ == matplotlib.patches.Circle:
                patch.remove()

        # show circles around the visible objects
        for object in objects_visible_to_human:
            circle = matplotlib.patches.Circle((object["x"], object["y"]), .5, edgecolor='blue', facecolor='none')
            self.ax_scatter_human.add_patch(circle)

        if text_labels:
            # remove previous text annotations
            for text in self.ax_scatter_human.texts:
                text.remove()

            # annotate objects by locations
            for o in robot_mm.dsg.objects:
                self.ax_scatter_human.text(robot_mm.dsg.objects[o]["x"], robot_mm.dsg.objects[o]["y"], str(robot_mm.dsg.objects[o]["class"] + " (" + str(round(robot_mm.dsg.objects[o]["x"],1)) + "," + str(round(robot_mm.dsg.objects[o]["y"],1)) + ")"), fontsize=8, color=(0,0,0,0.1))

        # update the rgb image
        self.plot_rgb_robot.set_data(rgb_robot[::-1,:,:])
        self.plot_rgb_human.set_data(rgb_human[::-1,:,:])

        # update the human trajectory view
        if human_trajectory_view is not None:
            self.plot_human_trajectory.set_data(np.rot90(human_trajectory_view[::-1,:,:], k=-1))

        self.text_frame.set_text(f"Frame: {frame_num}")

    def convert_continuous_to_plot(self, val):
        return
