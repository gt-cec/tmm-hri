# plot_full_tmm.py: plots several mental models scatter plots as well as the robot's RGB, segmentation, and depth images

import matplotlib.patches
import matplotlib.pyplot as plt
import numpy as np

class PlotFullTMM():
    def __init__(self, classes, class_to_class_id, class_id_to_color_map, use_gt_semantics):
        self.classes = classes
        self.class_to_class_id = class_to_class_id
        self.class_id_to_color_map = class_id_to_color_map
        self.use_gt_semantics = use_gt_semantics

        self.fig = plt.figure()
        self.fig.tight_layout()

        # scatter plot for robot mm
        self.ax_scatter_robot = self.fig.add_subplot(231)#, projection='3d')  # technically can be 3D, using 2D for simplicity
        self.ax_scatter_robot.set_xlim((-12, 12))
        self.ax_scatter_robot.set_ylim((-12, 12))
        self.ax_scatter_robot.set_aspect('equal')
        self.ax_scatter_robot.set_axis_off()
        self.ax_scatter_robot.set_title("Robot Semantic Map")
        xs, ys = [], []
        self.plot_scatter_robot = self.ax_scatter_robot.scatter(xs, ys, s=10)

        # scatter plot for predicted human mm
        self.ax_scatter_human = self.fig.add_subplot(232)#, projection='3d')  # technically can be 3D, using 2D for simplicity
        self.ax_scatter_human.set_xlim((-12, 12))
        self.ax_scatter_human.set_ylim((-12, 12))
        self.ax_scatter_human.set_aspect('equal')
        self.ax_scatter_human.set_axis_off()
        self.ax_scatter_human.set_title("Predicted Human Semantic Map")
        xs, ys = [], []
        self.plot_scatter_human = self.ax_scatter_human.scatter(xs, ys, s=10)

        # scatter plot for true human mm
        self.ax_scatter_human_true = self.fig.add_subplot(233)#, projection='3d')  # technically can be 3D, using 2D for simplicity
        self.ax_scatter_human_true.set_xlim((-12, 12))
        self.ax_scatter_human_true.set_ylim((-12, 12))
        self.ax_scatter_human_true.set_aspect('equal')
        self.ax_scatter_human_true.set_axis_off()
        self.ax_scatter_human_true.set_title("True Human Semantic Map")
        xs, ys = [], []
        self.plot_scatter_human_true = self.ax_scatter_human_true.scatter(xs, ys, s=10)

        # pose lines
        self.pose_lines = []

        # rgb view axis
        self.ax_rgb = self.fig.add_subplot(234)
        self.ax_rgb.set_xlim(0, 512)
        self.ax_rgb.set_ylim(0, 512)
        self.ax_rgb.set_aspect('equal')
        self.ax_rgb.set_axis_off()
        self.ax_rgb.set_title("Robot RGB Camera")
        self.plot_rgb = self.ax_rgb.imshow(np.zeros((512, 512, 3)))

        # segmentation view axis
        self.ax_seg = self.fig.add_subplot(235)
        self.ax_seg.set_xlim(0, 512)
        self.ax_seg.set_ylim(0, 512)
        self.ax_seg.set_aspect('equal')
        self.ax_seg.set_axis_off()
        self.ax_seg.set_title("Camera Segmentation")
        self.plot_seg = self.ax_seg.imshow(np.zeros((512, 512, 1)))

        # depth view axis
        self.ax_depth = self.fig.add_subplot(236)
        self.ax_depth.set_xlim(0, 512)
        self.ax_depth.set_ylim(0, 512)
        self.ax_depth.set_aspect('equal')
        self.ax_depth.set_axis_off()
        self.ax_depth.set_title("Robot Depth Camera")
        self.plot_depth = self.ax_depth.imshow(np.zeros((512, 512, 1)))

        # annotate the figure
        self.fig.text(0.01, 0.99, "Demo of Semantic Map Construction", ha='left', va='top', fontsize=14, color='black')
        classes_vert = 0.90
        self.fig.text(0.01, classes_vert, "Classes:", ha='left', va='top', fontsize=10, color='black')
        for i, c in enumerate(classes):
            classes_vert -= 0.025
            self.fig.text(0.01, classes_vert, "    ҉  " + c, ha='left', va='top', fontsize=8, color=class_id_to_color_map[i])
        self.text_frame = self.fig.text(0.99, 0.01, "Frame: N/A", ha="right", va="bottom", fontsize=10, color="black")


    def update(self, robot_mm, pred_human_mm, gt_human_mm, agent_pose, detected_objects, human_detections, objects_visible_to_human, rgb_image, depth_image, frame_num):
        # initialize the mental model scatter points
        mm_points_x_robot, mm_points_y_robot, mm_points_z_robot = [], [], []
        mm_points_x_human, mm_points_y_human, mm_points_z_human = [], [], []
        mm_points_x_human_true, mm_points_y_human_true, mm_points_z_human_true = [], [], []

        # add the robot agent
        mm_points_x_robot.append(agent_pose[0][0])  # right
        mm_points_y_robot.append(agent_pose[0][2])  # forward
        mm_points_z_robot.append(agent_pose[0][1])  # up
        plot_colors_robot, plot_colors_human, plot_colors_human_true = [(0, 0, 0, 1)], [], []  # plot colors of mm points, initialize the robot's with the agent pose

        # fade out the previous motion colors
        for i, line in enumerate(self.pose_lines):
            alpha = line.get_alpha()
            if alpha is not None:  # if the line has an alpha value
                if alpha <= 0.05:  # delete if going invisible
                    del self.pose_lines[i]
                    continue
                line.set_alpha(alpha - 0.05)  # decrease alpha

        # segment the RGB image
        seg = np.zeros((rgb_image.shape[0], rgb_image.shape[1], 4))  # segmentation colors
        for o in detected_objects:  # get the segmentation by color
            color = self.class_id_to_color_map[self.class_to_class_id[o["class"]]]
            seg[np.array(o["seg mask"]) > 0] = color

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

        if mm_points_x_robot != [] and mm_points_y_robot != [] and mm_points_z_robot != []:  # update the plot points and colors
            self.plot_scatter_robot.set_offsets(np.c_[mm_points_x_robot, mm_points_y_robot])
            self.plot_scatter_robot.set_color(plot_colors_robot)

        # if mm_points_x_human != [] and mm_points_y_human != [] and mm_points_z_human != []:  # update the plot points and colors
        self.plot_scatter_human.set_offsets(np.c_[mm_points_x_human, mm_points_y_human])
        self.plot_scatter_human.set_color(plot_colors_human)

        # if mm_points_x_human_true != [] and mm_points_y_human_true != [] and mm_points_z_human_true != []:  # update the plot points and colors
        self.plot_scatter_human_true.set_offsets(np.c_[mm_points_x_human_true, mm_points_y_human_true])
        self.plot_scatter_human_true.set_color(plot_colors_human_true)

        line = self.ax_scatter_robot.axes.plot((agent_pose[0][0], agent_pose[0][0] + agent_pose[-1][0]), (agent_pose[0][2], agent_pose[0][2] + agent_pose[-1][2]), color=(0, 0, 0), alpha=1.0)
        self.pose_lines.append(line[0])

        # update the rgb image
        self.plot_rgb.set_data(rgb_image[::-1,:,:])
        # plot_rgb.set_data(gt_semantic[::-1,:,:])
        for patch in self.ax_rgb.patches + self.ax_rgb.texts:  # remove existing rectangles
            patch.remove()
        for rgb_human in human_detections[0]:  # add rectangles for detected humans
            self.ax_rgb.add_patch(matplotlib.patches.Rectangle((rgb_human["box"][0][1], rgb_image.shape[1] - rgb_human["box"][0][0]), rgb_human["box"][1][1] - rgb_human["box"][0][1], rgb_human["box"][0][0] - rgb_human["box"][1][0], linewidth=1, edgecolor='r', facecolor='none'))
            self.ax_rgb.text(rgb_human["box"][0][0], rgb_image.shape[1] - rgb_human["box"][0][1], rgb_human["confidence"], fontfamily="sans-serif", fontsize=4, color="white", bbox=dict(facecolor="red", linewidth=1, alpha=1.0, edgecolor="red", pad=0))

        # remove previous patches (circles around visible objects)
        for patch in self.ax_scatter_human.get_children():
            if patch.__class__ == matplotlib.patches.Circle:
                patch.remove()

        # show circles around the visible objects
        for object in objects_visible_to_human:
            circle = matplotlib.patches.Circle((object["x"], object["y"]), .5, edgecolor='blue', facecolor='none')
            self.ax_scatter_human.add_patch(circle)

        # update the depth image
        self.plot_depth.set_data(depth_image[::-1,:,:] / 5)
        if not self.use_gt_semantics:  # ground truth semantics don't segment depth, so don't bother plotting it
            for patch in self.ax_depth.patches + self.ax_depth.texts:  # remove existing rectangles
                patch.remove()
            for depth_human in human_detections[1]:  # add rectangles for detected humans
                self.ax_depth.add_patch(matplotlib.patches.Rectangle((depth_human["box"][0][0], depth_image.shape[1] - depth_human["box"][0][1]), depth_human["box"][1][0] - depth_human["box"][0][0], depth_human["box"][0][1] - depth_human["box"][1][1], linewidth=1, edgecolor='r', facecolor='none'))
                self.ax_depth.text(depth_human["box"][0][0], depth_image.shape[1] - depth_human["box"][0][1], depth_human["confidence"], fontfamily="sans-serif", fontsize=4, color="white", bbox=dict(facecolor="red", linewidth=1, alpha=1.0, edgecolor="red", pad=0))

        # update the seg image
        self.plot_seg.set_data(seg[::-1,:,:])
        self.text_frame.set_text(f"Frame: {frame_num}")
