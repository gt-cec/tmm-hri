# logit_eval.py: evaluate the performance of the LLMs in identifying items for an activity on a logit-level comparison

import demo.identify_items_for_activity
import demo.qwen
import pickle

if __name__ == "__main__":
    print("Logit Evaluation: Identify Item Relevance for Activity")

    episode = "episode_42_short"
    print("Using episode:", episode)
    
    frames = [1, 100, 200, 300, 400, 500]
    print("Using frames:", frames)

    dist_limit = 0.3
    print("Using distance limit to flag unknown object:", dist_limit)

    # get all unknown objects at each of the given frames
    objects = demo.identify_items_for_activity.get_object_sets(episode, frames, dist_limit=dist_limit)

    # reduce to the set of unknown objects throughout all frames for an unbiased evaluation (i.e., no double evaluations)
    unknown_objects = set()
    for episode_frame in objects:
        [unknown_objects.add(x) for x in list(set(objects[episode_frame]["classes unknown to human"]))]
    
    print("All unknown objects:", ", ".join(unknown_objects))

    scores = {}
    for activity in demo.identify_items_for_activity.ACTIVITIES.keys():
        for obj in unknown_objects:
            print(f"Evaluating activity {activity} with object {obj}")

            # chain-of-thought prompt
            response, logits = demo.qwen.run(f"You are tasked with judging whether an object is relevant to an activity. You will be given an object and an activity, and you must return whether the object is relevant for the activity.\n\nFor example, if you are given an object and the activity is {activity}, return whether the object is relevant to the activity.\n\nYour turn! Is a {obj} useful for {activity}?")
            result = demo.qwen.eval_binary(logits)
            print("Identify via cot (", obj, activity, ") result:", result)
            with open("cot eval logits.txt", "a") as f:
                f.write(f"{obj}|{activity}|{result}\n")

            # single chain-of-thought prompt
            response, logits = demo.qwen.run(f"Is a {obj} useful for {activity}?")
            result = demo.qwen.eval_binary(logits)
            print("Identify via simple (", obj, activity, ") result:", result)
            with open("simple eval logits.txt", "a") as f:
                f.write(f"{obj}|{activity}|{result}\n")
    