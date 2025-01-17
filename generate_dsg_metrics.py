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
    dsg_files = [x for x in os.listdir(f"{episode_dir}/") if x.startswith("DSGs_")]
    for dsg_file in dsg_files:
        dsg_i = int(dsg_file.split("_")[1].split(".")[0])
        with open(f"{episode_dir}/{dsg_file}", "rb") as f:
            data["frames"][dsg_i] = pickle.load(f)

    # load the initial object state
    with open(f"{episode_dir}/episode_info.txt") as f:
        gt_semantic_colormap = ast.literal_eval(f.readlines()[3])
    classes = ["human", 'perfume', 'candle', 'bananas', 'cutleryfork', 'washingsponge', 'apple', 'cereal', 'lime', 'cellphone', 'bellpepper', 'crackers', 'garbagecan', 'chips', 'peach', 'toothbrush', 'pie', 'cupcake', 'creamybuns', 'plum', 'chocolatesyrup', 'towel', 'folder', 'toothpaste', 'computer', 'book', 'fryingpan', 'paper', 'mug', 'dishbowl', 'remotecontrol', 'dishwashingliquid', 'cutleryknife', 'plate', 'hairproduct', 'candybar', 'slippers', 'painkillers', 'whippedcream', 'waterglass', 'salmon', 'barsoap', 'character', 'wineglass']
    classes = sorted(list(set([gt_semantic_colormap[k][1] for k in gt_semantic_colormap if gt_semantic_colormap[k][1] in classes])) + ["human"])  # get the classes that are in the colormap
    graph = [{"class": gt_semantic_colormap[k][1], "x": gt_semantic_colormap[k][2][0], "y": gt_semantic_colormap[k][2][2], "z": gt_semantic_colormap[k][2][1]} for k in gt_semantic_colormap if gt_semantic_colormap[k][1] in classes]  # get the original objects
    objects_by_class = {}
    for node in graph:
        if node["class"] not in objects_by_class:
            objects_by_class[node["class"]] = []
        objects_by_class[node["class"]].append(node)
    data["initial"] = objects_by_class

    return data

def generate_dsg_metrics():
    data = load_dsg_data("episodes/episode_42")
    # generate the metrics
    similarities = {}
    for frame_id in data["frames"]:
        mm_robot = data["frames"][frame_id]["robot"]
        mm_human_gt = data["frames"][frame_id]["gt human"]
        mm_human_pred = data["frames"][frame_id]["pred human"]

        robot_set = format_objects_by_class(mm_robot.get_objects_by_class())
        human_gt_set = format_objects_by_class(mm_human_gt.get_objects_by_class())
        human_pred_set = format_objects_by_class(mm_human_pred.get_objects_by_class())
        initial_set = format_objects_by_class({class_name : data["initial"][class_name] for class_name in data["initial"] if class_name in robot_set})

        similarities[frame_id] = {
            "robot wrt initial": metrics.metrics.smcc(robot_set, initial_set),
            "human wrt initial": metrics.metrics.smcc(human_gt_set, initial_set),
            "pred wrt human": metrics.metrics.smcc(human_pred_set, human_gt_set),
            "pred wrt initial": metrics.metrics.smcc(human_pred_set, initial_set),
            "robot wrt human": metrics.metrics.smcc(robot_set, human_gt_set)
        }

        print(f"Frame {frame_id}: {similarities[frame_id]}")

    # plot the similarities
    frames = sorted(similarities.keys())
    plt.figure()
    plt.plot(frames, [similarities[frame_id]["robot wrt initial"] for frame_id in frames], label="Robot wrt Initial")
    plt.plot(frames, [similarities[frame_id]["robot wrt human"] for frame_id in frames], label="Robot wrt Human GT")
    plt.plot(frames, [similarities[frame_id]["human wrt initial"] for frame_id in frames], label="Human GT wrt Initial")
    plt.plot(frames, [similarities[frame_id]["pred wrt human"] for frame_id in frames], label="Pred wrt Human GT")
    plt.plot(frames, [similarities[frame_id]["pred wrt human"] - similarities[frame_id]["human wrt initial"] for frame_id in frames], label="Pred - Human GT")
    plt.xlabel("Frame")
    plt.ylabel("SMCC")
    plt.legend()
    plt.title("DSG Similarity Metrics")
    plt.grid()
    plt.tight_layout()
    plt.savefig("dsg_metrics.png")
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
    generate_dsg_metrics()