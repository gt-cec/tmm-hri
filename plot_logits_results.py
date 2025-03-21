# plot_logits_results.py: plots the results of evaluating truth based on logits

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import demo.identify_items_for_activity
import numpy as np

import demo.qwen

def create_violin_plot(data, ax, keys, title):
    # plot distribution for ground truth being true
    violins = ax.violinplot(data, showextrema=False)
    for pc in violins['bodies']:
        pc.set_linewidth(0.1)
    ax.set_xticks(range(len(keys) + 1))
    ax.set_xticklabels([""] + [x[0] + " " + x[1] for x in keys], rotation=45)
    ax.set_ylabel("Confidence (high is True)")
    ax.set_title(title)

    for i, points in enumerate(data):
        points = [x for x in points]
        jitter = np.random.uniform(-0.1, 0.1, size=len(points))
        x_positions = np.full_like(points, i + 1, dtype=float)
        ax.scatter(x_positions + jitter, points, alpha=0.6, color='grey', s=10)

def plot_logits_results(data, prompt):
    """
        Plots the results of evaluating truth based on logits as a 2D plot
    """  

    fig = plt.figure(figsize=(15, 5))
    fig.suptitle(f"Qwen2.5:3b Logits Results From: \"{prompt}\"")
    num_yesno_pairs = len(demo.qwen.token_ids) // 2
    gs = gridspec.GridSpec(2, num_yesno_pairs, figure=fig)#plt.subplots(2, 8, figsize=(15, 5))

    # Create the subplots
    axes = [[], []]
    # Make a subplot span multiple columns (e.g., ax4 will span across two columns)
    axes[0].append(fig.add_subplot(gs[0, 0 : num_yesno_pairs // 2]))
    axes[0].append(fig.add_subplot(gs[0, num_yesno_pairs // 2 : num_yesno_pairs]))
    for i in range(num_yesno_pairs):
        axes[1].append(fig.add_subplot(gs[1, i]))

    # set up the data
    data_by_logit = {}
    for activity in data.keys():
        for obj in data[activity].keys():
            ground_truth = obj in demo.identify_items_for_activity.ACTIVITIES[activity]
            for method in data[activity][obj]:
                if method not in data_by_logit:
                    data_by_logit[method] = []
                print(">>>", (ground_truth, data[activity][obj][method], obj, activity))
                data_by_logit[method].append((ground_truth, data[activity][obj][method], obj, activity))

    array_data_by_logit_gt_is_true = []
    for method in data_by_logit:
        array_data_by_logit_gt_is_true.append([x[1] for x in data_by_logit[method] if x[0] == True])

    array_data_by_logit_gt_is_false = []
    for method in data_by_logit:
        array_data_by_logit_gt_is_false.append([x[1] for x in data_by_logit[method] if x[0] == False])

    print("N (true) =", len(array_data_by_logit_gt_is_true[0]), "N (false) =", len(array_data_by_logit_gt_is_false[0]))
    
    create_violin_plot(array_data_by_logit_gt_is_true, axes[0][0], data_by_logit.keys(), f"Confidence of logits over True points (N = {len(array_data_by_logit_gt_is_true[0])})")
    create_violin_plot(array_data_by_logit_gt_is_false, axes[0][1], data_by_logit.keys(), f"Confidence of logits over False points (N = {len(array_data_by_logit_gt_is_false[0])})")

    # calculate true positive, false positive, true negative, false negative for each logit type
    summary_scores = {}
    false_negatives = {}

    for method in data_by_logit:
        summary_scores[method] = {"TP": 0, "FP": 0, "TN": 0, "FN": 0}
        false_negatives[method] = []
        for i in range(len(data_by_logit[method])):
            if data_by_logit[method][i][0] == True:
                if data_by_logit[method][i][1] > 0.5:
                    summary_scores[method]["TP"] += 1
                else:
                    summary_scores[method]["FN"] += 1
                    false_negatives[method].append(data_by_logit[method][i])
            else:
                if data_by_logit[method][i][1] > 0.5:
                    summary_scores[method]["FP"] += 1
                else:
                    summary_scores[method]["TN"] += 1
    
    print("False negatives:", false_negatives)
    for i, key in enumerate(summary_scores):
        # print(key, summary_scores[key])
        # Add first table
        axes[1][i].axis('off')
        axes[1][i].set_title(f"{key[0]} {key[1]}")
        table = axes[1][i].table(cellText=[[summary_scores[key]["TP"], summary_scores[key]["FN"]], [summary_scores[key]["FP"], summary_scores[key]["TN"]]], colLabels=["T", "F"], rowLabels=[" T ", " F "], loc='upper left', cellLoc='center', colColours=['#f0f0f0']*3)
        # Shade the row labels
        for (row, col), cell in table.get_celld().items():
            if (row == 0) or (col == -1):
                cell.set_facecolor('#D3D3D3')  # Set background color (light gray)
                cell.set_text_props(weight='bold')  # Set text color (black)
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 1.5)
        axes[1][i].annotate('Predicted', xy=(0.5, 1.01), xycoords='axes fraction', ha='center', va='center', fontsize=9)
        axes[1][i].annotate('Actual', xy=(-.25, 0.7), rotation=90, xycoords='axes fraction', ha='center', va='center', fontsize=9)

        F1 = round(2 * summary_scores[key]["TP"] / (2 * summary_scores[key]["TP"] + summary_scores[key]["FP"] + summary_scores[key]["FN"]), 2)
        precision = round(summary_scores[key]["TP"] / (summary_scores[key]["TP"] + summary_scores[key]["FP"]) if summary_scores[key]["TP"] + summary_scores[key]["FP"] > 0 else 0, 2)
        recall = round(summary_scores[key]["TP"] / (summary_scores[key]["TP"] + summary_scores[key]["FN"]), 2)
        axes[1][i].annotate(f"F1: {F1}\nPrec: {precision}\nRecall: {recall}", (0, 0.5), (0, 0.5), xycoords='axes fraction', textcoords='offset points', va='top')

    plt.tight_layout()
    plt.show()
    
def load_text(filename):
    """
    Loads the text from a file
    """
    data = {}

    with open(filename, "r") as f:
        lines = f.readlines()
        for line in lines:
            items = line.split("|")
            obj = items[0]
            activity = items[1]
            value = eval(items[2])
            if activity not in data:
                data[activity] = {}
            data[activity][obj] = value
    return data


data = load_text("cot eval logits.txt")
# plot the data
plot_logits_results(data, "Intro, symbolic example, Is a _ useful for _?")