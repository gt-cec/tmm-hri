import matplotlib.pyplot as plt
import matplotlib.patches
import metrics.metrics
import os, pickle, utils, ast, statistics

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


def generate_dsg_smcc_plot(episode_dir:str, ablation_annotation=None):
    data = load_dsg_data(episode_dir)
    # generate the metrics
    similarities = {}
    prev_robot_set = None
    for frame_id in sorted(data["frames"].keys()):
        dsg_robot = data["frames"][frame_id]["robot"]
        dsg_human_gt = data["frames"][frame_id]["gt human"]
        dsg_human_pred = data["frames"][frame_id]["pred human"]

        robot_set = format_objects_by_class(dsg_robot.get_objects_by_class())
        human_gt_set = format_objects_by_class(dsg_human_gt.get_objects_by_class())
        human_pred_set = format_objects_by_class(dsg_human_pred.get_objects_by_class())
        initial_set = format_objects_by_class({class_name : data["initial"][class_name] for class_name in data["initial"] if class_name in robot_set})

        similarities[frame_id] = {
            "robot wrt human": metrics.metrics.smcc(robot_set, human_gt_set),
            "robot wrt initial": metrics.metrics.smcc(robot_set, initial_set),
            "human wrt initial": metrics.metrics.smcc(human_gt_set, initial_set),
            "pred wrt human": metrics.metrics.smcc(human_pred_set, human_gt_set),
            "pred wrt initial": metrics.metrics.smcc(human_pred_set, initial_set)
        }

        if prev_robot_set is not None:
            metrics.metrics.smcc(robot_set, prev_robot_set)
        prev_robot_set = robot_set

    # plot the similarities
    frames = sorted(similarities.keys())
    plt.figure()
    
    # remove border
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    
    # plot the similarities
    plot_pred_wrt_human = plt.plot(frames, [similarities[frame_id]["pred wrt human"] for frame_id in frames], label="★[Pred wrt Human GT] Distance between the inferred and the human's scene graph.", color="#ff1e6b")
    plot_pred_wrt_initial = plt.plot(frames, [similarities[frame_id]["pred wrt initial"] for frame_id in frames], label="[Pred wrt Initial] Error of the inferred human scene graph.", color="#ff1e6b", linestyle=":")
    plot_robot_wrt_human = plt.plot(frames, [similarities[frame_id]["robot wrt human"] for frame_id in frames], label="[Robot wrt Human GT] Distance between the robot's and the human's scene graph.", color="#1eb6ff")
    plot_robot_wrt_initial = plt.plot(frames, [similarities[frame_id]["robot wrt initial"] for frame_id in frames], label="[Robot wrt Initial] Error of the robot's scene graph.", color="#1eb6ff", linestyle=":")
    plot_human_wrt_initial = plt.plot(frames, [similarities[frame_id]["human wrt initial"] for frame_id in frames], label="[Human GT wrt Initial] Error of the human's scene graph.", color="#2e8600", linestyle=":")
   
    # set the labels
    axis_fontsize = 15
    plt.title("DSG Similarity Metrics: Parents are Out", fontsize=15)
    plt.xlabel("Frame", fontsize=axis_fontsize)
    plt.xlim([frames[0], frames[-1]])
    plt.ylabel("Mean SMCC [m]", fontsize=axis_fontsize)
    plt.ylim([0, 2])

    # add the shading for the human is seen
    for frame_id in frames:
        # add a verticle rectangle if the human was seen in this frame
        if data["frames"][frame_id]["human is seen"]:
            plt.gca().add_patch(plt.Rectangle((frame_id - 0.5, 0), 1, 2, facecolor="purple", alpha=0.1))

    # create the grid
    plt.grid()
    
    # create the legend
    handles = [
        plot_pred_wrt_human[0],  # [0] to get the label component
        plot_pred_wrt_initial[0],
        plot_robot_wrt_human[0],
        plot_robot_wrt_initial[0],
        plot_human_wrt_initial[0],
        matplotlib.patches.Patch(facecolor='purple', alpha=0.1, label='Human is observed in this frame')
    ]
    legend = plt.legend(handles=handles, framealpha=1, fontsize=7.5, loc="upper left", bbox_to_anchor=(0, 1.02))
    legend.get_frame().set_linewidth(0)

    # add the annotation
    if ablation_annotation is not None:
        add_annotation(plt.gca(), ablation_annotation)

    # draw a downward arrow on the y-axis to indicate lower is better
    x_pos = -0.107
    plt.annotate("", xy=(x_pos, 0.2), xytext=(x_pos, 0), xycoords="axes fraction", arrowprops=dict(arrowstyle="<-", lw=0.5, color="black"))

    plt.tight_layout()
    plt.savefig("dsg_metrics.png", dpi=500)
    plt.show()


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
            dsg_human_pred = data[description]["frames"][frame_id]["pred human"]

            robot_set = format_objects_by_class(dsg_robot.get_objects_by_class())
            human_gt_set = format_objects_by_class(dsg_human_gt.get_objects_by_class())
            human_pred_set = format_objects_by_class(dsg_human_pred.get_objects_by_class())
            initial_set = format_objects_by_class({class_name : data[description]["initial"][class_name] for class_name in data[description]["initial"] if class_name in robot_set})

            similarities[description][frame_id] = {
                "robot wrt human": metrics.metrics.smcc(robot_set, human_gt_set),
                "robot wrt initial": metrics.metrics.smcc(robot_set, initial_set),
                "human wrt initial": metrics.metrics.smcc(human_gt_set, initial_set),
                "pred wrt human": metrics.metrics.smcc(human_pred_set, human_gt_set),
                "pred wrt initial": metrics.metrics.smcc(human_pred_set, initial_set)
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
    generate_dsg_metrics("episodes/episode_42")
    # generate_dsg_smcc_plot(episode_dir="episodes/episode_42", ablation_annotation="Ablation: GT all but pose")