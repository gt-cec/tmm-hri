# llm.py: wrapper for using an LLM to classify semantic relationships between objects and activities
import langchain
import pydantic
import langchain_core.prompts
import langchain_core.output_parsers
import langchain_ollama.llms
import langchain.output_parsers
import demo.qwen

# class for the LLM demo
class LLMDemoActivityNeeds:
    """LLM Demo Activity Needs."""

    # Pydantic class for the candidate items to tell the user
    class CandidateItems(pydantic.BaseModel):
        """List of items that the user may need."""
        relevant_items: list[str] = pydantic.Field(description="The items relevant to the activity.")

    def __init__(self, NONE_OBJECT="None of these objects"):
        """Initialize the LLM demo."""
        self.NONE_OBJECT = NONE_OBJECT

        # set up the LLM
        model = langchain_ollama.llms.OllamaLLM(model="qwen2.5:32b-instruct", temperature=0)

        # list chain-of-thought: give the set of objects to the LLM and ask for the subset that is relevant for the activity
        # set up the prompt
        self.list_cot_output_parser = langchain_core.output_parsers.JsonOutputParser(pydantic_object=self.CandidateItems)
        list_cot_prompt = langchain_core.prompts.PromptTemplate(
            input_variables=["belief_state", "activity"],
            template="You are tasked with identifying objects that are relevant to an activity. You will be given a set of objects and an activity, and you must return the subset that is relevant for the activity.\n\nFor example, if you are given the objects [A, B, C, D, E] and the objects A, C, D are relevant to the task, return {{relevant_items: ['A', 'B', 'C']}}.\n\nYour turn! You are aware of the following items:\n{belief_state}\nWhich items are useful for {activity}?\n\nReturn your output as a JSON object with the given structure.",
            output_parser=self.list_cot_output_parser
        )
        # set up the chain
        self.list_cot_chain = list_cot_prompt | model

        # judge chain-of-thought: give a single object to the LLM and ask for a judgment on whether it is relevant for the activity
        # set up the prompt
        self.single_cot_output_parser = langchain.output_parsers.boolean.BooleanOutputParser()
        single_cot_prompt = langchain_core.prompts.PromptTemplate(
            input_variables=["object", "activity"],
            template="You are tasked with judging whether an object is relevant to an activity. You will be given an object and an activity, and you must return whether the object is relevant for the activity.\n\nFor example, if you are given an object and the activity is {activity}, return `true` if the object is relevant and `false` if it is not.\n\nYour turn! Is a {object} useful for {activity}?\n\nReturn your output as `true` or `false`.",
            output_parser=self.single_cot_output_parser
        )
        # set up the chain
        self.single_cot_chain = single_cot_prompt | model

        # simple judge yes or no: give a single object to the LLM and ask for a judgment on whether it is relevant for the activity
        # set up the prompt
        self.single_judge_output_parser = langchain.output_parsers.boolean.BooleanOutputParser()
        single_judge_prompt = langchain_core.prompts.PromptTemplate(
            input_variables=["object", "activity"],
            template="Is a {object} useful for {activity}? Answer `true` or `false` with no other explanation.",
            output_parser=self.single_judge_output_parser
        )
        # set up the chain
        self.single_judge_chain = single_judge_prompt | model

    def identify_via_list_cot(self, belief_state: list[str], activity:str):
        """Identify the items that the user may need."""
        response = self.list_cot_chain.invoke({"belief_state": "\n".join(["- " + x for x in belief_state + [self.NONE_OBJECT]]), "activity": activity})
        subset = self.list_cot_output_parser.parse(response)
        print("Identify via list cot:", response, subset)
        return subset["relevant_items"]
    
    def identify_via_single_cot(self, objects: list[str], activity:str, use_hf:bool=True) -> list[str]:
        """Identify the relevant items using a chain of thought prompt."""
        selected = []
        for obj in objects:
            if use_hf:
                response, logits = demo.qwen.run(f"You are tasked with judging whether an object is relevant to an activity. You will be given an object and an activity, and you must return whether the object is relevant for the activity.\n\nFor example, if you are given an object and the activity is {activity}, return whether the object is relevant to the activity.\n\nYour turn! Is a {obj} useful for {activity}?")
                result = demo.qwen.eval_binary(logits)
                print("Identify via HUGGINGFACE single cot (", obj, activity, ") result:", result)
                with open("cot judge logits.txt", "a") as f:
                    f.write(f"judge {obj} {activity} {result}\n")
                # if result[("true", "false")] > 0.5:
                #     selected.append(obj)
            else:
                result = self.single_cot_chain.invoke({"object": obj, "activity": activity})
                print("Identify via single cot (", obj, activity, ") result:", result)
                if result.lower().strip().replace(".", "").split(" ")[0] == "true":
                    selected.append(obj)
        if len(selected) == 0:
            selected = [self.NONE_OBJECT]
        return selected
    
    def identify_via_single_judge(self, objects: list[str], activity:str, use_hf:bool=True) -> list[str]:
        """Identify the relevant items using a short judge prompt."""
        selected = []
        for obj in objects:
            if use_hf:
                response, logits = demo.qwen.run(f"Is a {obj} useful for {activity}?")
                result = demo.qwen.eval_binary(logits)
                print("Identify via HUGGINGFACE single judge (", obj, activity, ") result:", result)
                with open("single judge logits.txt", "a") as f:
                    f.write(f"judge {obj} {activity} {result}\n")
                # if result[("true", "false")] > 0.5:
                #     selected.append(obj)
            else:
                result = self.single_judge_chain.invoke({"object": obj, "activity": activity})
                print("Identify via single judge:", result)
                if result.lower().strip().replace(".", "").split(" ")[0] == "true":
                    selected.append(obj)
        if len(selected) == 0:
            selected = [self.NONE_OBJECT]
        return selected
