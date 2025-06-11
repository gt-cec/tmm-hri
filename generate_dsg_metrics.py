import metrics.metrics
import os, pickle, statistics

episode_numbers = [100, 101, 102, 103, 105, 106, 107, 108, 109, 110]  # seed 104 led to issues with navigation to objects

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

def load_dsg_data(episode_dir:str, description:str="") -> dict:
    """
    Load the DSG data from the saved files.
    :return: The DSG data.
    """
    data = {"frames": {}}

    # load the DSG data
    subfolder = "DSGs" + (" " + description if description != "" else "")
    # if the subfolder does not exist, try to load the data from the inference folder
    if not os.path.exists(f"{episode_dir}/{subfolder}"):
        subfolder = subfolder.replace("with ", "")
    if not os.path.exists(f"{episode_dir}/{subfolder}"):
        subfolder = subfolder.replace("Inference", "Path")
    if not os.path.exists(f"{episode_dir}/{subfolder}"):
        raise FileNotFoundError(f"Subfolder {subfolder} does not exist in {episode_dir}")
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
        mean_similarities[description] = {}
        std_similarities[description] = {}
        values_by_category = {}
        for frame_id in similarities[description]:
            for category in similarities[description][frame_id]:
                if category not in mean_similarities[description]:
                    mean_similarities[description][category] = []
                    std_similarities[description][category] = []
                if category not in values_by_category:
                    values_by_category[category] = []
                values_by_category[category].append(similarities[description][frame_id][category])

        for category in values_by_category:
            mean_similarities[description][category] = statistics.mean(values_by_category[category])
            std_similarities[description][category] = statistics.stdev(values_by_category[category])

    # print the metrics categorized by k
    for k in similarities[descriptions[0]][frame_id]:
        print(f"\n{k}")
        for description in descriptions:
            print(f"{description}:\t{round(mean_similarities[description][k], 3)} ± {round(std_similarities[description][k], 3)}")
    return mean_similarities, std_similarities, len(data[descriptions[0]]["frames"])

def calculate_overall_mean_std(results):
    """
    Calculate the overall mean and std of the pred wrt human results.
    :param results: The results to calculate the mean and std for.
    :return: The overall mean and std.
    """
    overall_mean = {}
    overall_variance = {}
    category = "pred wrt human"
    for description in descriptions:
        means_across_episodes = [results[episode][description]["means"][category][0] for episode in results if description in results[episode]]
        stds_across_episodes = [results[episode][description]["stds"][category][0] for episode in results if description in results[episode]]
        ns_across_episodes = [results[episode][description]["n"][category] for episode in results if description in results[episode]]
        # calculate weighted average
        overall_mean[description] = sum(m * n for m, n in zip(means_across_episodes, ns_across_episodes)) / sum(ns_across_episodes)
        # calculate pooled variance
        overall_variance[description] = sum((n - 1) * (s ** 2) + n * (m - overall_mean[description]) ** 2 for s, n, m in zip(stds_across_episodes, ns_across_episodes, means_across_episodes)) / (sum(ns_across_episodes) - len(ns_across_episodes))
    print("!!", overall_mean, overall_variance)
    return overall_mean, overall_variance

if __name__ == "__main__":
    csv_string = ""

    # load the overall results if the pkl exists
    if os.path.exists("overall_results.pkl"):
        with open("overall_results.pkl", "rb") as f:
            overall_results = pickle.load(f)
        print("Loaded overall results from overall_results.pkl")
        overall_mean, overall_variance = calculate_overall_mean_std(overall_results)
        for description in descriptions:
            print(f"{description}:\t{round(overall_mean[description], 3)} ± {round(overall_variance[description] ** 0.5, 3)}")

    else:
        overall_results = {}
        for episode_number in episode_numbers:
            means, stds, n = generate_dsg_metrics(f"episodes/episode_{episode_number}")
            overall_results[episode_number] = {}
            for description in descriptions:
                if description not in overall_results:
                    overall_results[episode_number][description] = {"means": {}, "stds": {}, "n": {}}
                for k in means[description]:
                    if k not in overall_results[episode_number][description]["means"]:
                        overall_results[episode_number][description]["means"][k] = []
                        overall_results[episode_number][description]["stds"][k] = []
                    overall_results[episode_number][description]["means"][k].append(means[description][k])
                    overall_results[episode_number][description]["stds"][k].append(stds[description][k])
                    overall_results[episode_number][description]["n"][k] = n
            if csv_string == "":
                csv_string += f"Episode Number"
                for description in descriptions:
                    for k in means[description]:
                        csv_string += f", {description} {k} mean, {description} {k} std"
            csv_string += f"\n{episode_number}"
            for description in descriptions:
                for k in means[description]:
                    csv_string += f", {means[description][k]}, {stds[description][k]}"

            # save the overall results to a file
            if episode_number == episode_numbers[-1]:
                with open("overall_results.pkl", "wb") as f:
                    pickle.dump(overall_results, f)
    
    print(csv_string)
    print("Done!")
