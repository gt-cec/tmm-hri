import matplotlib.pyplot as plt
import metrics.metrics
import os, pickle, utils, ast

def load_dsg_data(episode_dir:str) -> dict:
    """
    Load the DSG data from the saved files.
    :return: The DSG data.
    """
    data = {"frames": {}}

    # load the DSG data
    dsg_files = [x for x in os.listdir(f"{episode_dir}/DSGs") if x.startswith("DSGs_")]
    for dsg_file in dsg_files:
        dsg_i = int(dsg_file.split("_")[1].split(".")[0])
        with open(f"{episode_dir}/DSGs/{dsg_file}", "rb") as f:
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

def generate_dsg_metrics(episode_dir:str):
    data = load_dsg_data(f"{episode_dir}")
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
    plt.plot(frames, [similarities[frame_id]["pred wrt human"] for frame_id in frames], label="★[Pred wrt Human GT] Distance between the inferred and the human's scene graph.", color="#ff1e6b")
    plt.plot(frames, [similarities[frame_id]["pred wrt initial"] for frame_id in frames], label="[Pred wrt Initial] Drift of the inferred human scene graph.", color="#ff1e6b", linestyle=":")
    plt.plot(frames, [similarities[frame_id]["robot wrt human"] for frame_id in frames], label="[Robot wrt Human GT] Distance between the robot's and the human's scene graph.", color="#1eb6ff")
    plt.plot(frames, [similarities[frame_id]["robot wrt initial"] for frame_id in frames], label="[Robot wrt Initial] Drift of the robot's scene graph.", color="#1eb6ff", linestyle=":")
    plt.plot(frames, [similarities[frame_id]["human wrt initial"] for frame_id in frames], label="[Human GT wrt Initial] Drift of the human's scene graph.", color="#2e8600", linestyle=":")
    # set the labels
    axis_fontsize = 15
    plt.title("DSG Similarity Metrics: Walking About the House", fontsize=15)
    plt.xlabel("Frame", fontsize=axis_fontsize)
    plt.xlim([frames[0], frames[-1]])
    plt.ylabel("Mean SMCC [m]", fontsize=axis_fontsize)
    plt.ylim([0, 0.30])
    # create the grid
    plt.grid()
    # create the legend
    legend = plt.legend(framealpha=1, fontsize=7.5, loc="upper left", bbox_to_anchor=(0, 1.02))
    legend.get_frame().set_linewidth(0)
    # place the legend at the top of the plot
    # plt.legend(loc="upper center", bbox_to_anchor=(0.5, 1.15), ncol=3)

    plt.tight_layout()
    plt.savefig("dsg_metrics.png", dpi=500)
    plt.show()

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

if __name__ == "__main__":
    generate_dsg_metrics(episode_dir="episodes/episode_42")