import matplotlib.pyplot as plt
import metrics.metrics
import os, pickle

def load_dsg_data(episode_dir:str) -> dict:
    """
    Load the DSG data from the saved files.
    :return: The DSG data.
    """
    data = {}
    dsg_files = [x for x in os.listdir(f"{episode_dir}/") if x.startswith("DSGs_")]
    for dsg_file in dsg_files:
        dsg_i = int(dsg_file.split("_")[1].split(".")[0])
        with open(f"{episode_dir}/{dsg_file}", "rb") as f:
            data[dsg_i] = pickle.load(f)
    return data

def generate_dsg_metrics():
    data = load_dsg_data("episodes/episode_42")
    # generate the metrics
    for frame_id in data:
        mm_robot = data[frame_id]["robot"]
        mm_human_gt = data[frame_id]["gt human"]
        mm_human_pred = data[frame_id]["pred human"]

        robot_set = format_objects_by_class(mm_robot.get_objects_by_class())
        human_gt_set = format_objects_by_class(mm_human_gt.get_objects_by_class())
        human_pred_set = format_objects_by_class(mm_human_pred.get_objects_by_class())

        similarity = metrics.metrics.smcc(human_pred_set, human_gt_set)

        print(f"Frame {frame_id}: {similarity}")

    # Show the plots
    plt.show()

def format_objects_by_class(objects_by_class:dict) -> list:
    """
    Format the objects by class to be a list of lists.
    :param objects_by_class: The objects by class.
    :return: The formatted objects by class.
    """
    formatted_objects_by_class = []
    for class_name in objects_by_class:
        formatted_objects_by_class.append([(x["x"], x["y"]) for x in objects_by_class[class_name]])
    return formatted_objects_by_class

if __name__ == "__main__":
    generate_dsg_metrics()