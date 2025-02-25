s = """
llm list cot F1: 0.647887323943662
llm list cot TP: 23
llm list cot FP: 22
llm list cot TN: 159
llm list cot FN: 3
llm single cot F1: 0.6666666666666666
llm single cot TP: 18
llm single cot FP: 10
llm single cot TN: 171
llm single cot FN: 8
llm single judge F1: 0.5416666666666666
llm single judge TP: 13
llm single judge FP: 9
llm single judge TN: 172
llm single judge FN: 13
bert F1: 0.6756756756756757
bert TP: 25
bert FP: 23
bert TN: 158
bert FN: 1
random F1: 0.20077220077220076
random TP: 13.0
random FP: 90.5
random TN: 90.5
random FN: 13.0
"""

# calculate the precision and recall for each method
scores = {}
for line in s.strip().split("\n"):
    parts = line.split()
    method = parts[:-2]
    method = " ".join(method)
    metric = parts[-2].replace(":", "")
    value = float(parts[-1])
    if method not in scores:
        scores[method] = {}
    scores[method][metric] = value

for method in scores:
    print("- Method:", method)
    scores[method]["precision"] = scores[method]["TP"] / (scores[method]["TP"] + scores[method]["FP"]) if scores[method]["TP"] + scores[method]["FP"] > 0 else 0
    scores[method]["recall"] = scores[method]["TP"] / (scores[method]["TP"] + scores[method]["FN"]) if scores[method]["TP"] + scores[method]["FN"] > 0 else 0
    print(f"{method} F1: {scores[method]['F1']}")
    print(f"{method} precision: {scores[method]['precision']}")
    print(f"{method} recall: {scores[method]['recall']}")

print(scores)
