# runs the demos for the project

import demo.identify_items_for_activity
import pickle

if __name__ == "__main__":
    print("- Demo: Identify Items for Activity -")

    episode = "episode_42"
    print("Using episode:", episode)

    scores = {}
    for activity in demo.identify_items_for_activity.ACTIVITIES.keys():
        for frame_id in [1, 100, 200, 300, 400, 500]:
            print(f"Running demo for activity {activity} at frame {frame_id}")
            scores = demo.identify_items_for_activity.run_demo(episode, activity, dist_limit=0.3, frame_id=frame_id, verbose=True)    

    summary_scores = {
        "llm list cot": {"TP": 0, "FP": 0, "TN": 0, "FN": 0},
        "llm single cot": {"TP": 0, "FP": 0, "TN": 0, "FN": 0},
        "llm single judge": {"TP": 0, "FP": 0, "TN": 0, "FN": 0},
        "bert": {"TP": 0, "FP": 0, "TN": 0, "FN": 0},
        "random": {"TP": 0, "FP": 0, "TN": 0, "FN": 0}
    }

    # set the random scores to llm single cot, should be the same across all methods
    for activity in scores:
        summary_scores["random"]["TP"] += sum(scores[activity]["llm single cot"]["random TP"])
        summary_scores["random"]["FP"] += sum(scores[activity]["llm single cot"]["random FP"])
        summary_scores["random"]["TN"] += sum(scores[activity]["llm single cot"]["random TN"])
        summary_scores["random"]["FN"] += sum(scores[activity]["llm single cot"]["random FN"])

    for activity in scores:
        for method in summary_scores.keys():
            if method == "random":
                continue
            summary_scores[method]["TP"] += sum(scores[activity][method]["TP"])
            summary_scores[method]["FP"] += sum(scores[activity][method]["FP"])
            summary_scores[method]["TN"] += sum(scores[activity][method]["TN"])
            summary_scores[method]["FN"] += sum(scores[activity][method]["FN"])
    
    for method in summary_scores.keys():
        summary_scores[method]["F1"] = 2 * summary_scores[method]["TP"] / (2 * summary_scores[method]["TP"] + summary_scores[method]["FP"] + summary_scores[method]["FN"])
        print(f"{method} F1:", summary_scores[method]["F1"])
        print(f"{method} TP:", summary_scores[method]["TP"])
        print(f"{method} FP:", summary_scores[method]["FP"])
        print(f"{method} TN:", summary_scores[method]["TN"])
        print(f"{method} FN:", summary_scores[method]["FN"])

    # Total random TP: 13.0
    # Total random FP: 90.5
    # Total random TN: 90.5
    # Total random FN: 13.0
    # Random F1: 0.20077220077220076

    with open(f"demo_results_{episode}.pkl", "wb") as f:
        pickle.dump(scores, f)