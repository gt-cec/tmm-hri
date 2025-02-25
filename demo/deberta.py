from transformers import AutoModelForSequenceClassification, DebertaV2Tokenizer, pipeline

class DeBERTav3:
    def __init__(self, model_name="MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli"):
        self.classifier = pipeline("zero-shot-classification", model="MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli")
        self.tokenizer = DebertaV2Tokenizer.from_pretrained(model_name)

    # predicts whether an object used for an activity is logical or illogical
    def predict_one(self, object, activity):
        sequence_to_classify = f"I am {activity} using a {object}."
        candidate_labels = ["A sentence that makes sense.", "A sentence that does not make sense."]
        output = self.classifier(sequence_to_classify, candidate_labels, multi_label=False)
        return output["labels"][0] == candidate_labels[0]

    # predicts whether a list of objects used for an activity is logical or illogical
    def predict_many(self, objects, activity):
        selected = []
        for obj in objects:
            logical = self.predict_one(activity, obj)
            print(f"Using a {obj} for {activity} is {logical}")
            if logical:
                selected.append(obj)
        if len(selected) == 0:
            selected = ["None of these objects"]
        return selected

if __name__ == "__main__":
    bert = DeBERTav3()
    print(bert.predict_many(["remote control", "toothbrush", "dish bowl", "cutlery fork", "washing sponge", "bar soap", "towel"], "washing the dishes"))