from transformers import AutoModelForCausalLM, AutoTokenizer
import math

model_name = "Qwen/Qwen2.5-3B-Instruct"

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

token_ids = {
    "true": 1866,
    "false": 3849,
    "True": 2514,
    "False": 4049,
    "TRUE": 20611,
    "FALSE": 30351,
    "=true": 11265,
    "=false": 12219,
    "=True": 3618,
    "=False": 5608,
    "yes": 9693,
    "no": 2152,
    "Yes": 9454,
    "No": 2753,
    "YES": 14004,
    "NO": 8996,
    "verity": 26472,
    "nay": 98073
}

def run(prompt):
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt}
    ]
    print("MESSAGES", messages)
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=1,
        output_logits=True,
        return_dict_in_generate=True
    )

    _generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids["sequences"])
    ]

    response = tokenizer.batch_decode(_generated_ids, skip_special_tokens=False)[0]
    return response, generated_ids["logits"]

def eval_binary(logits):
    ratios = {}
    token_keys = list(token_ids.keys())
    for i in range(0, len(token_keys), 2):
        true_weight = float(logits[0][0][token_ids[token_keys[i]]])
        false_weight = float(logits[0][0][token_ids[token_keys[i+1]]])
        softmax = math.exp(true_weight) / (math.exp(true_weight) + math.exp(false_weight))
        ratios[(token_keys[i], token_keys[i+1])] = softmax
        print("First Logit:", token_keys[i], ":", true_weight, token_keys[i+1], ":", false_weight, "Softmax", softmax, "True Ratio", true_weight / (true_weight + false_weight) if (true_weight + false_weight) > 0 else "???" + str(true_weight) + " " + str(false_weight), "False Ratio", false_weight / (true_weight + false_weight) if (true_weight + false_weight) > 0 else "???" + str(true_weight) + " " + str(false_weight))
    return ratios