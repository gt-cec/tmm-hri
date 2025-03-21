# identify_items_for_activity.py: uses the mental model to identify information to tell the user

# see the README.md for the instructions to run this code
import os, pickle
import dsg
import utils

NONE_OBJECT = "None of these objects"

# VirtualHome classes lack spaces, add them
OBJECT_NAME_LOOKUP = {
    "washingsponge": "washing sponge",
    "dishbowl": "dish bowl",
    "dishwashingliquid": "dishwashing liquid",
    "wineglass": "wine glass",
    "cutleryknife": "cutlery knife",
    "cutleryfork": "cutlery fork",
    "fryingpan": "frying pan",
    "remotecontrol": "remote control",
    "barsoap": "bar soap"
}

# researcher-crafted ground truth, activity name to relevant objects
ACTIVITIES = {
    "cleaning the window": [
        "washing sponge",
    ],
    "washing the dishes": [
        "dish bowl",
        "dishwashing liquid",
        "washing sponge",
        "plate",
        "mug",
        "wine glass",
        "towel",
        "cutlery fork",
        "cutlery knife"
    ],
    "making a sandwich": [
        "plate",
        "cutlery knife"
    ],
    "cooking a meal": [
        "pot",
        "frying pan",
        "plate",
        "cutlery fork",
        "cutlery knife",
        "apple",
        "peach",
        "plum",
        "cupcake",
        "dish bowl"
    ],
    "watching television": [
        "remote control"
    ],
    "reading a book": [
        "book"
    ],
    "doing laundry": [
        "towel"
    ],
    "taking a shower": [
        "bar soap",
        "towel"
    ],
    "brushing teeth": [
        "toothbrush",
        "toothpaste"
    ]
}

scores = {}

# initialize the model
model = None
bert = None

def load_models():
    import demo.deberta
    import demo.llm
    global model, bert
    model = demo.llm.LLMDemoActivityNeeds(NONE_OBJECT=NONE_OBJECT)
    bert = demo.deberta.DeBERTav3()
    
def load_latest_dsgs_from_episode(episode:str):
    """Load the latest dynamic scene graphs from an episode.""" 
    # get the DSG file with the highest ID
    dsg_files = os.listdir(f"episodes/{episode}/DSGs")
    dsg_files.sort()
    frame_id = dsg_files[-1].split("_")[1].split(".")[0]
    return load_dsgs_from_episode(episode, frame_id)


def load_dsgs_from_episode(episode:str, frame_id:str):
    """Load a dynamic scene graph from an episode."""
    # check if the frame exists
    if not os.path.exists(f"episodes/{episode}/DSGs/DSGs_{frame_id}.pkl"):
        raise ValueError(f"Specified frame does not exist! Frame: {frame_id}")
    with open(f"episodes/{episode}/DSGs/DSGs_{frame_id}.pkl", "rb") as f:
        return pickle.load(f)
    

def parse_object_classes(dsg:dsg) -> list[str]:
    """Parse the object classes from a dynamic scene graph."""
    return list(set([OBJECT_NAME_LOOKUP[dsg.objects[obj]['class']] if dsg.objects[obj]['class'] in OBJECT_NAME_LOOKUP else dsg.objects[obj]['class'] for obj in dsg.objects]))


def filter_dsg_to_objects_beyond_dist(dsg_source:dsg, dsg_target:dsg, dist_limit=3) -> dsg:
    """Filter a dynamic scene graph to only include objects. Uses the source DSG 
    as the baseline and gets objects in the target DSG that do not have a class close
    enough to a source object of that class."""

    # filter the target objects to only include objects that are not close to a source object of that class
    unknown_objects = []
    distances = {}
    for target_obj in {k:v for k, v in dsg_target.objects.items()}:
        # get the class of the object
        target_class = dsg_target.objects[target_obj]["class"]
        target_loc = (dsg_target.objects[target_obj]["x"], dsg_target.objects[target_obj]["y"])
        # get the closest object of this class in the source DSG
        min_dist = float("inf")
        for source_obj in dsg_source.objects:
            if dsg_source.objects[source_obj]["class"] != target_class:
                continue
            source_loc = (dsg_source.objects[source_obj]["x"], dsg_source.objects[source_obj]["y"])
            dist = utils.dist(target_loc, source_loc)
            if dist < min_dist:
                min_dist = dist

        # record the distance if far enough away, otherwise remove it from the DSG
        if min_dist > dist_limit:
            distances[target_obj] = round(min_dist, 3)
        else:
            dsg_target.remove_object(target_obj)
            
    # resulting DSG only has objects far enough away from the source
    return dsg_target, distances

def get_object_sets(episode:str, frame_ids:list[int], dist_limit=0.3):
    """Get the object sets from the DSGs of the specified episodes and frame IDs."""
    object_sets = {}
    for frame_id in frame_ids:
        dsgs = load_dsgs_from_episode(episode, frame_id)
        key = f"{episode}_{frame_id}"
        if key not in object_sets:
            object_sets[key] = {}
        object_sets[key]["classes known to robot"] = parse_object_classes(dsgs["robot"])
        # get the DSGs of objects not in the robot's DSG
        dsgs_objects_human_does_not_know, distances = filter_dsg_to_objects_beyond_dist(dsg_source=dsgs["robot"], dsg_target=dsgs["pred human"], dist_limit=dist_limit)
        
        # parse the object classes
        objects_human = parse_object_classes(dsgs_objects_human_does_not_know)
        object_sets[key]["classes unknown to human"] = objects_human
    return object_sets

def run_demo(episode:str, activity:str="cooking", dist_limit:float=0.3, frame_id:int=-1, verbose:bool=False):
    """Run the demo for identifying items for an activity that the human is not aware of.
    The demo uses the saved dynamic scene graphs from a VirtualHome episode and compares
    the predicted human scene graph against the robot's scene graph."""

    # load the DSGs
    if frame_id == -1:
        dsgs = load_latest_dsgs_from_episode(episode)
    else:
        dsgs = load_dsgs_from_episode(episode, frame_id)

    # get the DSGs of objects not in the robot's DSG
    dsgs_objects_human_does_not_know, distances = filter_dsg_to_objects_beyond_dist(dsg_source=dsgs["robot"], dsg_target=dsgs["pred human"], dist_limit=dist_limit)
    
    # parse the object classes
    objects_human = parse_object_classes(dsgs_objects_human_does_not_know)
    
    # identify the items that the user may need
    # llm_list_cot_proposed = model.identify_via_list_cot(objects_human, activity)
    llm_single_cot_proposed = model.identify_via_single_cot(objects_human, activity, use_hf=True)
    print("OBJECTS", objects_human)
    llm_single_judge_proposed = model.identify_via_single_judge(objects_human, activity, use_hf=True)
    # bert_proposed = bert.predict_many(objects_human, activity)

    # grade_demo("llm list cot", objects_human, activity, llm_list_cot_proposed)
    grade_demo("llm single cot", objects_human, activity, llm_single_cot_proposed)
    grade_demo("llm single judge", objects_human, activity, llm_single_judge_proposed)
    # grade_demo("bert", objects_human, activity, bert_proposed)

    if verbose:
        print(f"Objects far enough from the robot's belief state: {objects_human}")
        print(f"  Distances: {distances}")
        # print(f"  Objects deemed relevant to {activity} by list COT: {llm_list_cot_proposed}")
        print(f"  Objects deemed relevant to {activity} by single COT: {llm_single_cot_proposed}")
        print(f"  Objects deemed relevant to {activity} by single judge: {llm_single_judge_proposed}")
        # print(f"  Objects deemed relevant to {activity} by DeBERTa: {bert_proposed}")
    
    return scores


def grade_demo(method:str, objects_human:list[str], activity:str, proposed_items:list[str]):
    # copy the activities because we will modify it
    relevant_objects = ACTIVITIES[activity]

    # if there are no objects in the activity that are in objects_human, set it to NONE_OBJECT
    relevant_objects = [x for x in relevant_objects if x in objects_human]
    if len(relevant_objects) == 0:
        relevant_objects = [NONE_OBJECT]

    # for each item in the activity, check if it is in the proposed items
    correct = []
    incorrect = []
    missed = []
    correct_not = []
    TP, FP, TN, FN = 0, 0, 0, 0
    max_score = 0
    for item in objects_human:
        # true positive
        if item in proposed_items and item in relevant_objects:
            correct.append(item)
            TP += 1
        # false positive
        elif item in proposed_items and item not in relevant_objects:
            incorrect.append(item)
            FP += 1
        # true negative
        elif item not in proposed_items and item not in relevant_objects:
            correct_not.append(item)
            TN += 1
        # false negative
        elif item not in proposed_items and item in relevant_objects:
            missed = []
            FN += 1

    # calculate the F1 score
    precision = TP / (TP + FP) if TP + FP > 0 else 0
    recall = TP / (TP + FN) if TP + FN > 0 else 0
    F1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0

    # random
    relevant_objects_in_objects_human = [x for x in relevant_objects if x in objects_human]
    rand_tp = len(relevant_objects_in_objects_human) * 0.5  # expect to get half of the relevant objects correctly affirmed
    rand_tn = (len(objects_human) - len(relevant_objects_in_objects_human)) * 0.5  # expect to get half of the irrelevant objects correctly rejected
    rand_fp = (len(objects_human) - len(relevant_objects_in_objects_human)) * 0.5  # expect to get half of the irrelevant objects incorrectly affirmed
    rand_fn = len(relevant_objects_in_objects_human)* 0.5  # expect to get half of the relevant objects incorrectly rejected

    # calculate the F1 score
    precision = rand_tp / (rand_tp + rand_fp) if rand_tp + rand_fp > 0 else 0
    recall = rand_tp / (rand_tp + rand_fn) if rand_tp + rand_fn > 0 else 0
    random_F1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0

    # calculate the score if the user randomly guessed
    if activity not in scores:
        scores[activity] = {}

    if method not in scores[activity]:
        scores[activity][method] = {
            "TP": [],
            "FP": [],
            "TN": [],
            "FN": [],
            "correct": [],
            "incorrect": [],
            "missed": [],
            "random TP": [],
            "random FP": [],
            "random TN": [],
            "random FN": []
        }

    scores[activity][method]["TP"].append(TP)
    scores[activity][method]["FP"].append(FP)
    scores[activity][method]["TN"].append(TN)
    scores[activity][method]["FN"].append(FN)
    scores[activity][method]["random TP"].append(rand_tp)
    scores[activity][method]["random FP"].append(rand_fp)
    scores[activity][method]["random TN"].append(rand_tn)
    scores[activity][method]["random FN"].append(rand_fn)
    scores[activity][method]["correct"].append(correct)
    scores[activity][method]["incorrect"].append(incorrect)
    scores[activity][method]["missed"].append(missed)
    return scores
