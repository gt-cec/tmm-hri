import matplotlib.pyplot as plt
import matplotlib.patches
import metrics.metrics
import os, pickle, utils, ast, statistics

# plot colors
# color_sunset_orange = (252/255, 112/255, 35/255)
# color_sunset_redorange = (250/255, 51/255, 41/255)
# color_sunset_brown = (164/255, 27/255, 31/255)
color_blue_dark = (20/255, 67/255, 148/255)
color_redpurple = (161/255, 17/255, 28/255)
color_blue_medium = (15/255, 126/255, 253/255, 0)
color_blue_light = (130/255, 165/255, 214/255)
color_black = (0, 0, 0)

color_inferred = color_redpurple #color_blue_dark
color_robot = color_black #color_blue_light
color_human = color_blue_medium

color_human_is_seen = color_blue_dark
color_human_is_seen_alpha = 0#0.25  # set to zero while plotting the other lines

# plot parameters
ablation_annotation = None  # comment out this to show the ablation annotation
axis_fontsize = 15
legend_adjustment = (0.5, -0.12)
legend_fontsize = 13
ylim_max = 1.4
whiteout_y_start = 1.4
arrow_x_pos = -0.11
arrow_start_y = 0.15

def load_dsg_data(episode_dir:str, description:str="") -> dict:
    """
    Load the DSG data from the saved files.
    :return: The DSG data.
    """
    data = {"frames": {}}

    # load the DSG data
    subfolder = "DSGs" + (" " + description if description != "" else "")
    dsg_files = [x for x in os.listdir(f"{episode_dir}/{subfolder}") if x.startswith("DSGs_")]
    for dsg_file in dsg_files:
        dsg_i = int(dsg_file.split("_")[1].split(".")[0])
        with open(f"{episode_dir}/{subfolder}/{dsg_file}", "rb") as f:
            data["frames"][dsg_i] = pickle.load(f)

    # load the initial object state
    with open(f"{episode_dir}/starting_graph.pkl", "rb") as f:
        graph = pickle.load(f)
    classes = ["human", 'perfume', 'candle', 'bananas', 'cutleryfork', 'washingsponge', 'apple', 'cereal', 'lime', 'cellphone', 'bellpepper', 'crackers', 'garbagecan', 'chips', 'peach', 'toothbrush', 'pie', 'cupcake', 'creamybuns', 'plum', 'chocolatesyrup', 'towel', 'folder', 'toothpaste', 'computer', 'book', 'fryingpan', 'paper', 'mug', 'dishbowl', 'remotecontrol', 'dishwashingliquid', 'cutleryknife', 'plate', 'hairproduct', 'candybar', 'slippers', 'painkillers', 'whippedcream', 'waterglass', 'salmon', 'barsoap', 'character', 'wineglass']
    objects_by_class = {}
    for node in graph["nodes"]:
        if node["class_name"] not in classes:
            continue
        if node["class_name"] not in objects_by_class:
            objects_by_class[node["class_name"]] = []
        objects_by_class[node["class_name"]].append({"class": node["class_name"], "x": node["bounding_box"]["center"][0], "y": node["bounding_box"]["center"][2], "z": node["bounding_box"]["center"][1]})
    data["initial"] = objects_by_class
    return data


def generate_dsg_smcc_plot(ax, episode_dir:str, description:str="", ablation_annotation=None, show_y_label=True, hide_everything=False):
    data = load_dsg_data(episode_dir, description)
    # generate the metrics
    similarities = {}
    prev_robot_set = None
    for frame_id in sorted(data["frames"].keys()):
        dsg_robot = data["frames"][frame_id]["robot"]
        dsg_human_gt = data["frames"][frame_id]["gt human"]
        dsg_human_inferred = data["frames"][frame_id]["pred human"]

        robot_set = format_objects_by_class(dsg_robot.get_objects_by_class())
        human_gt_set = format_objects_by_class(dsg_human_gt.get_objects_by_class())
        human_inferred_set = format_objects_by_class(dsg_human_inferred.get_objects_by_class())
        initial_set = format_objects_by_class({class_name : data["initial"][class_name] for class_name in data["initial"] if class_name in robot_set})

        similarities[frame_id] = {
            "robot wrt human": metrics.metrics.smcc(robot_set, human_gt_set),
            "robot wrt initial": metrics.metrics.smcc(robot_set, initial_set),
            "human wrt initial": metrics.metrics.smcc(human_gt_set, initial_set),
            "pred wrt human": metrics.metrics.smcc(human_inferred_set, human_gt_set),
            "pred wrt initial": metrics.metrics.smcc(human_inferred_set, initial_set)
        }

        if prev_robot_set is not None:
            metrics.metrics.smcc(robot_set, prev_robot_set)
        prev_robot_set = robot_set

    # plot the similarities
    frames = sorted(similarities.keys())
    # plt.figure()
    
    # remove border
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # plot the similarities
    timesteps = [x / 10 for x in frames]
    plot_inferred_wrt_human = ax.plot(timesteps, [similarities[frame_id]["pred wrt human"] for frame_id in frames], label="★[Inferred vs. Human] Error of the inferred belief state.", color=color_inferred)
    plot_robot_wrt_initial = ax.plot(timesteps, [similarities[frame_id]["robot wrt initial"] for frame_id in frames], label="[Robot vs. True] Error of the robot's belief state.", color=color_robot, linestyle="-")
    plot_human_wrt_initial = ax.plot(timesteps, [similarities[frame_id]["human wrt initial"] for frame_id in frames], label="[Human vs. True] Error of the human's belief state.", color=color_human, linestyle="-")
   
    # set the labels
    xticks = [i for i in range(0, int(timesteps[-1]), 10)]
    ax.set_xlabel("Time [s]", fontsize=axis_fontsize)
    ax.set_xticks(ticks=xticks, labels=xticks, fontsize=legend_fontsize)
    ax.set_xlim([timesteps[0], timesteps[-1]])

    ytick_interval = 0.2
    num_yticks = int(ylim_max / 0.2) + 2
    yticks = [round(ytick_interval * i, 1) for i in range(0, num_yticks)]
    ax.set_title(ablation_annotation, fontsize=axis_fontsize, fontweight="bold")
    if show_y_label:
        ax.set_ylabel("Mean SMCC [m]", fontsize=axis_fontsize)
        # draw a downward arrow on the y-axis to indicate lower is better
        ax.annotate("", xy=(arrow_x_pos, arrow_start_y), xytext=(arrow_x_pos, 0), xycoords="axes fraction", arrowprops=dict(arrowstyle="<-", lw=0.5, color="black"))
    ax.set_yticks(ticks=yticks, labels=yticks, fontsize=legend_fontsize)
    ax.set_ylim([0, ylim_max])

    if not hide_everything:
        # add the shading for the human is seen
        for i in range(len(frames)):
            frame_id = frames[i]
            timestep = timesteps[i]
            # add a verticle rectangle if the human was seen in this frame
            if data["frames"][frame_id]["human is seen"]:
                width = 0.1066
                ax.add_patch(plt.Rectangle((timestep - width / 2, 0), width, 2, linewidth=0, facecolor=color_human_is_seen, alpha=color_human_is_seen_alpha))

    # create the grid
    if not hide_everything:
        ax.grid()  # hiding while plotting the others

    return plot_inferred_wrt_human, plot_robot_wrt_initial, plot_human_wrt_initial

# utility function for retrieving the visible frames, use before add_visible_frames to fix a DSG folder that doesn't have the "human is seen" flag
def get_visible_frames(episode_dir:str, description:str=""):
    data = load_dsg_data(episode_dir, description)
    visible_frames = {}
    # add the shading for the human is seen
    for frame_id in data["frames"]:
        # add a verticle rectangle if the human was seen in this frame
        visible_frames[frame_id] = data["frames"][frame_id]["human is seen"]
    
    filename = "visible_frames.pkl"
    with open(filename, "wb") as f:
        pickle.dump(visible_frames, f)
    
    print(f"Done obtaining the visible frames, saved to {filename}")
    print(visible_frames)

# utility function for adding the visible frames to a DSG folder, use after get_visible_frames to fix a DSG folder that doesn't have the "human is seen" flag
def add_visible_frames(episode_dir:str, description:str=""):
    # load the visible frames
    filename = "visible_frames.pkl"
    with open(filename, "rb") as f:
        visible_frames = pickle.load(f)

    # load the DSG data
    subfolder = "DSGs" + (" " + description if description != "" else "")
    dsg_files = [x for x in os.listdir(f"{episode_dir}/{subfolder}") if x.startswith("DSGs_")]
    for dsg_file in dsg_files:
        dsg_i = int(dsg_file.split("_")[1].split(".")[0])
        with open(f"{episode_dir}/{subfolder}/{dsg_file}", "rb") as f:
            data = pickle.load(f)
            data["human is seen"] = visible_frames[dsg_i]
        
        # save the DSG data
        with open(f"{episode_dir}/{subfolder}/{dsg_file}", "wb") as f:
            pickle.dump(data, f)

    print("Done adding the visible frames in to", episode_dir, description)


def add_annotation(ax, text):
    """
    Add an annotation to the plot at the bottom left

    Args:
        ax: axis to add the annotation to
        text: text to add
    """
    ax.annotate(text, xy=(-0.1, -0.11), xycoords='axes fraction', fontsize=12, color="red", ha='left', va='center')


def format_objects_by_class(objects_by_class:dict) -> list:
    """
    Format the objects by class to be a list of lists.
    :param objects_by_class: The objects by class.
    :return: The formatted objects by class.
    """
    formatted_objects_by_class = {}
    for class_name in objects_by_class:
        if class_name == "character":
            continue
        formatted_objects_by_class[class_name] = [(x["x"], x["y"]) for x in objects_by_class[class_name]]
    return formatted_objects_by_class


def generate_dsg_metrics(episode_dir:str):
    descriptions = [
        "Parents are Out GT Robot GT Human",
        "Parents are Out GT Robot with AStar Inference GT Human",
        "Parents are Out GT Robot with No Inference GT Human",
        "Parents are Out GT Robot with Online Detect GT Human",
        "Parents are Out GT Robot with Online Pose GT Human",
        "Parents are Out Online Robot GT Human",
        "Parents are Out Online Robot with GT Detect GT Human",
        "Parents are Out Online Robot with GT Inference GT Human",
        "Parents are Out Online Robot with GT Pose GT Human",
        "Parents are Out Online Robot with No Inference GT Human",
    ]

    data = {k : load_dsg_data(episode_dir, k) for k in descriptions}

    # generate the metrics
    similarities = {}
    for description in descriptions:
        similarities[description] = {}
        prev_robot_set = None
        for frame_id in sorted(data[description]["frames"].keys()):
            dsg_robot = data[description]["frames"][frame_id]["robot"]
            dsg_human_gt = data[description]["frames"][frame_id]["gt human"]
            dsg_human_inferred = data[description]["frames"][frame_id]["pred human"]

            robot_set = format_objects_by_class(dsg_robot.get_objects_by_class())
            human_gt_set = format_objects_by_class(dsg_human_gt.get_objects_by_class())
            human_inferred_set = format_objects_by_class(dsg_human_inferred.get_objects_by_class())
            initial_set = format_objects_by_class({class_name : data[description]["initial"][class_name] for class_name in data[description]["initial"] if class_name in robot_set})

            similarities[description][frame_id] = {
                "robot wrt human": metrics.metrics.smcc(robot_set, human_gt_set),
                "robot wrt initial": metrics.metrics.smcc(robot_set, initial_set),
                "human wrt initial": metrics.metrics.smcc(human_gt_set, initial_set),
                "pred wrt human": metrics.metrics.smcc(human_inferred_set, human_gt_set),
                "pred wrt initial": metrics.metrics.smcc(human_inferred_set, initial_set)
            }

            if prev_robot_set is not None:
                metrics.metrics.smcc(robot_set, prev_robot_set)
            prev_robot_set = robot_set
    
    # calculate the mean and std of the metrics
    mean_similarities = {}
    std_similarities = {}
    for description in descriptions:
        mean_similarities[description] = {k: statistics.mean([similarities[description][frame_id][k] for frame_id in similarities[description]]) for k in similarities[description][frame_id]}
        std_similarities[description] = {k: statistics.stdev([similarities[description][frame_id][k] for frame_id in similarities[description]]) for k in similarities[description][frame_id]}

    # print the metrics categorized by k
    for k in similarities[descriptions[0]][frame_id]:
        print(f"\n{k}")
        for description in descriptions:
            print(f"{description}:\t{round(mean_similarities[description][k], 3)} ± {round(std_similarities[description][k], 3)}")
    

if __name__ == "__main__":
    generate_two = False
    show_legend = False
    hide_everything=False

    # GENERATE SIDE BY SIDE
    if generate_two:    
        fig, (ax_online, ax_gt) = plt.subplots(1, 2, squeeze=True)
        fig.set_size_inches(13.5, 5)
        # Note: GT Robot Online Human is the reverse
        plot_online_inferred_wrt_human, plot_online_robot_wrt_initial, plot_online_human_wrt_initial = generate_dsg_smcc_plot(ax_online, episode_dir="episodes/episode_42_short", description="Parents are Out Online Robot GT Human", ablation_annotation="")
        plot_gt_inferred_wrt_human, plot_gt_robot_wrt_initial, plot_gt_human_wrt_initial = generate_dsg_smcc_plot(ax_gt, episode_dir="episodes/episode_42_short", description="Parents are Out GT Robot GT Human", ablation_annotation="", show_y_label=False)

    # GENERATE ONE
    if not generate_two:
        fig, (ax) = plt.subplots(1, 1, squeeze=True)
        fig.set_size_inches(6.75, 5)
        # Note: GT Robot Online Human is the reverse
        plot_online_inferred_wrt_human, plot_online_robot_wrt_initial, plot_online_human_wrt_initial = generate_dsg_smcc_plot(ax, episode_dir="episodes/episode_42_short", description="Parents are Out Online Robot GT Human", ablation_annotation="", hide_everything=hide_everything)
        # plot_gt_inferred_wrt_human, plot_gt_robot_wrt_initial, plot_gt_human_wrt_initial = generate_dsg_smcc_plot(ax, episode_dir="episodes/episode_42_short", description="Parents are Out GT Robot GT Human", ablation_annotation="", show_y_label=False)

    # create the legend
    if show_legend:
        handles = [
            plot_online_inferred_wrt_human[0],  # [0] to get the label component
            matplotlib.patches.Patch(facecolor=color_human_is_seen, alpha=color_human_is_seen_alpha, label='The human is observed in this frame.'),
            plot_online_human_wrt_initial[0],
            plot_online_robot_wrt_initial[0]
        ]
        legend = fig.legend(handles=handles, framealpha=1, fontsize=legend_fontsize, loc="lower center", bbox_to_anchor=legend_adjustment, ncol=2)
        legend.get_frame().set_linewidth(0)
    
    plt.tight_layout()
    
    plt.savefig("parents are out baseline.svg", bbox_inches="tight", pad_inches=0, transparent=True)
    plt.show()