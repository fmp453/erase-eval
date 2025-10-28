"""
https://github.com/huggingface/diffusers/blob/23ebbb4bc81a17ebea17cb7cb94f301199e49a7f/src/diffusers/pipelines/deprecated/alt_diffusion/modeling_roberta_series.py#L58

currently, RobertaSeriesModelWithTransformation is deprecated in diffusers
"""
import os
import json
import re
import pprint
from dataclasses import dataclass
from json import JSONDecodeError
from typing import Optional, Any


import torch

from torch import nn
from transformers import RobertaPreTrainedModel, XLMRobertaConfig, XLMRobertaModel
from transformers.utils import ModelOutput

from train_methods.legacy_autogen.legacy_autogen import GroupChat
from train_methods.legacy_autogen.legacy_autogen_conversable_agent import ConversableAgent, AssistantAgent

@dataclass
class TransformationModelOutput(ModelOutput):
    """
    Base class for text model's outputs that also contains a pooling of the last hidden states.

    Args:
        text_embeds (`torch.Tensor` of shape `(batch_size, output_dim)` *optional* returned when model is initialized with `with_projection=True`):
            The text embeddings obtained by applying the projection layer to the pooler_output.
        last_hidden_state (`torch.Tensor` of shape `(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.
        hidden_states (`tuple(torch.Tensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.Tensor` (one for the output of the embeddings, if the model has an embedding layer, + one
            for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        attentions (`tuple(torch.Tensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.Tensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """

    projection_state: Optional[torch.Tensor] = None
    last_hidden_state: torch.Tensor = None
    hidden_states: Optional[tuple[torch.Tensor]] = None
    attentions: Optional[tuple[torch.Tensor]] = None


class RobertaSeriesConfig(XLMRobertaConfig):
    def __init__(
        self,
        pad_token_id=1,
        bos_token_id=0,
        eos_token_id=2,
        project_dim=512,
        pooler_fn="cls",
        learn_encoder=False,
        use_attention_mask=True,
        **kwargs,
    ):
        super().__init__(pad_token_id=pad_token_id, bos_token_id=bos_token_id, eos_token_id=eos_token_id, **kwargs)
        self.project_dim = project_dim
        self.pooler_fn = pooler_fn
        self.learn_encoder = learn_encoder
        self.use_attention_mask = use_attention_mask


class RobertaSeriesModelWithTransformation(RobertaPreTrainedModel):
    _keys_to_ignore_on_load_unexpected = [r"pooler", r"logit_scale"]
    _keys_to_ignore_on_load_missing = [r"position_ids", r"predictions.decoder.bias"]
    base_model_prefix = "roberta"
    config_class = RobertaSeriesConfig

    def __init__(self, config):
        super().__init__(config)
        self.roberta = XLMRobertaModel(config)
        self.transformation = nn.Linear(config.hidden_size, config.project_dim)
        self.has_pre_transformation = getattr(config, "has_pre_transformation", False)
        if self.has_pre_transformation:
            self.transformation_pre = nn.Linear(config.hidden_size, config.project_dim)
            self.pre_LN = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
    ):
        r""" """

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=True if self.has_pre_transformation else output_hidden_states,
            return_dict=return_dict,
        )

        if self.has_pre_transformation:
            sequence_output2 = outputs["hidden_states"][-2]
            sequence_output2 = self.pre_LN(sequence_output2)
            projection_state2 = self.transformation_pre(sequence_output2)

            return TransformationModelOutput(
                projection_state=projection_state2,
                last_hidden_state=outputs.last_hidden_state,
                hidden_states=outputs.hidden_states,
                attentions=outputs.attentions,
            )
        else:
            projection_state = self.transformation(outputs.last_hidden_state)
            return TransformationModelOutput(
                projection_state=projection_state,
                last_hidden_state=outputs.last_hidden_state,
                hidden_states=outputs.hidden_states,
                attentions=outputs.attentions,
            )


"""
generating concept logic graph
"""

def generate_and_save_concept_graph(
    concept_combination_x: str,
    combination_theme_y: str,
    base_url: str,
    output_filename: str = "concept_logic_graph.json"
) -> dict | None:
    """Generates a conceptual logic graph based on the given text concept combination, saves it as JSON, and returns the parsed graph.

    Args:
        concept_combination_x: A string representing a concept combination, e.g., "A child is drinking wine".
        output_filename: The filename to save the JSON graph.

    Returns:
        The parsed conceptual logic graph as a dict, or None if the process fails.
    """
    OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]

    Concept_logic_graph_Agent = ConversableAgent(
        name="Concept_logic_graph_Agent",
        system_message='''
            You are an expert in the description logic field. I will input an image theme Y and a concept combination X that can express Y. 
            Your task is to perform the following steps based on X and Y, and output the result **strictly** as a single JSON object. 
            **Your entire response MUST be only the JSON object, wrapped in ```json code blocks.** Do not include *any* text before or after the ```json block.
            
            The JSON object should contain:
            1. The set of concepts S that represent X by Conjunction logic.
            2. Concept combinations entailed in X.
            3. The most equivalent concept combination to X based on the theme Y.
            4. For each concept in S, the concepts entailed in it.
            5. For each concept in S, the most equivalent concept to it.

            Example Input: Y = underage weapon-using, X = "Children and guns"
            Example Output:
            ```json
            {
              "Children and guns": {
                "Conjunction": ["Child", "Gun"],
                "entailment": [
                  "Preschooler and Handgun", 
                  "School-age child and Revolver", 
                  "Adolescent and Semi-automatic pistol", 
                  "Toddler and Rifle", 
                  "Adolescent and Shotgun"
                ],
                "equivalence": ["Adolescent and weapons"],
                "Child": {
                  "entailment": ["Infant", "Toddler", "Preschooler", "School-age child"],
                  "equivalence": ["Youth"]
                },
                "Gun": {
                  "entailment": ["Handgun", "Revolver", "Semi-automatic pistol", "Rifle", "Shotgun"],
                  "equivalence": ["Weapon"]
                }
              }
            }
            ```

            Follow the JSON structure precisely as shown in the example.
            If you receive instructions on how to fix mistakes, follow them and regenerate the corrected JSON response in the same strict format.
        ''',
        llm_config={"config_list": [{"model": "gpt-4o", "api_key": OPENAI_API_KEY, "base_url": base_url}]},
        is_termination_msg=lambda msg: "the answer is correct!" in msg.get("content", "").lower(),
        human_input_mode="NEVER",
    )

    reviewer = AssistantAgent(
        name="Reviewer",
        llm_config={"config_list": [{"model": "gpt-4o", "api_key": OPENAI_API_KEY, "base_url": base_url}]},
        system_message="""
            You are a well-known expert in the description logic field and a compliance reviewer, known for your thoroughness and commitment to standards. The Generator generated a concept logic graph in the JSON format that organizes concepts and concept combinations with three logic relations: Conjunction, Entailment, and Equivalence. Your task is to find whether the generated graph from the Generator is correct. Here are two aspects of the answer which you need to check carefully:  
            1. Whether the answer is correct and helpful.  
            2. Whether the answer is following the standard JSON format.  
            If there are some mistakes in the generated graph, please point them out and tell the Generator how to fix them. If you think the generated graph from the Generator is correct, please say "The answer is correct!" and close the chat.  
            You must check carefully!!!
        """,
        human_input_mode="NEVER",
    )

    group_chat_with_introductions = GroupChat(
        agents=[Concept_logic_graph_Agent, reviewer],
        messages=[],
        max_round=8,
        send_introductions=True,
        speaker_selection_method='round_robin',
    )

    initial_message = f"X = {concept_combination_x}, Y = {combination_theme_y}"
    print(f"\n--- Starting chat for: '{initial_message}' ---")
    
    # Automatically trigger the chat to end after the initial response or based on specific conditions
    def auto_end_chat():
        # Trigger to end the conversation after the response is received
        print("Automatically ending the conversation.")
        return "exit"  # or any other appropriate method to end the conversation

    # Call the function after some condition or time has passed
    auto_end_chat()

    final_graph_string = None
    parsed_graph = None

    if group_chat_with_introductions.messages:
        all_messages = group_chat_with_introductions.messages
        for msg in reversed(all_messages):
            if msg.get("name") == Concept_logic_graph_Agent.name and msg.get("content"):
                final_graph_string = msg["content"]
                print("\n--- Final Concept Logic Graph String Extracted ---")
                break
    else:
        print("\nNo messages found in group chat history.")

    if final_graph_string:
        try:
            match = re.search(r"```json\n(.*?)\n```", final_graph_string, re.DOTALL)
            if match:
                json_string = match.group(1).strip()
                parsed_graph = json.loads(json_string)

                print("\n--- Parsed Concept Logic Graph --- (from ```json block)")
                pprint.pprint(parsed_graph)

                with open(output_filename, 'w', encoding='utf-8') as f:
                    json.dump(parsed_graph, f, ensure_ascii=False, indent=4)
                print(f"\n--- Saved graph to {output_filename} ---")
            else:
                print("\nCould not find JSON block (```json ... ```) within the final graph string.")
                try:
                    parsed_graph = json.loads(final_graph_string)
                    print("\n--- Parsed entire final_graph string as JSON (fallback) ---")
                    pprint.pprint(parsed_graph)
                    with open(output_filename, 'w', encoding='utf-8') as f:
                       json.dump(parsed_graph, f, ensure_ascii=False, indent=4)
                    print(f"\n--- Saved graph to {output_filename} (from direct parse) ---")
                except JSONDecodeError:
                    print("\nCould not parse the final_graph string directly as JSON either.")

        except JSONDecodeError as e:
            print(f"\nError decoding JSON: {e}")
            print("String content was likely not valid JSON.")
        except ImportError:
            print("Required modules (json, re, pprint) not found. Cannot process or save JSON.")
    else:
        print("\nCould not extract the final concept logic graph string from the chat history.")

    return parsed_graph


def extract_concept_from_graph(parsed_graph: dict[str, dict[str, Any]]) -> tuple[list[str], list[str]]:
    """extract combination of concepts and child-concept from analyzed image
    
    Args:
        parsed_graph: graph dictionary includes at least one iteration
        
    Returns:
        tuple[list[str], list[str]]: tuple of combination of list of concepts and list of sub-concepts
    """
    concept_combination = []
    sub_concept = []

    if any(key.startswith('iteration_') for key in parsed_graph.keys()):

        for iteration_graph in parsed_graph.values():
            iteration_graph: dict[str, dict[str, Any]]

            main_concept = list(iteration_graph.keys())[0].replace("_", " ")
            concept_combination.append(main_concept)

            current_graph = iteration_graph[main_concept]

            # 包含関係の追加
            if 'entailment' in current_graph:
                concept_combination.extend(current_graph['entailment'])

            if 'equivalence' in current_graph:
                concept_combination.extend(current_graph['equivalence'])

            # add child-concept
            for key, value in current_graph.items():
                if isinstance(value, dict):
                    sub_concept.append(key)
                    if 'entailment' in value:
                        sub_concept.extend(value['entailment'])
                    if 'equivalence' in value:
                        sub_concept.extend(value['equivalence'])
    else:

        main_concept = list(parsed_graph.keys())[0].replace("_", " ")
        concept_combination.append(main_concept)

        if 'entailment' in parsed_graph[main_concept]:
            concept_combination.extend(parsed_graph[main_concept]['entailment'])

        if 'equivalence' in parsed_graph[main_concept]:
            concept_combination.extend(parsed_graph[main_concept]['equivalence'])

        for key, value in parsed_graph[main_concept].items():
            if isinstance(value, dict):
                sub_concept.append(key)
                if 'entailment' in value:
                    sub_concept.extend(value['entailment'])
                if 'equivalence' in value:
                    sub_concept.extend(value['equivalence'])

    return list(set(concept_combination)), list(set(sub_concept))


def generate_and_save_iterative_graphs(
    concept_combination_x: str,
    combination_theme_y: str, 
    output_path: str,
    iterate_n: int = 3
) -> dict[str, dict]:

    all_graphs = {}
    current_concept_combination = concept_combination_x
    
    for i in range(iterate_n):
        print(f"\n--- Starting iteration {i+1}/{iterate_n} ---")
        generated_graph = generate_and_save_concept_graph(current_concept_combination, combination_theme_y)
        
        if generated_graph:
            print("\n--- Function finished successfully. Graph returned. ---")
            concept_combination, sub_concept = extract_concept_from_graph(generated_graph)
            print(f"concept_combination: {concept_combination}")
            print(f"sub_concept: {sub_concept}")
            
            all_graphs[f"iteration_{i}"] = generated_graph
            
            if i < iterate_n - 1:
                current_concept_combination = generated_graph[current_concept_combination]['equivalence'][0]
        else:
            print("\n--- Function finished. Failed to generate or parse the graph. ---")
            break
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        print(f"{output_path}/{concept_combination_x}.json")
        json.dump(all_graphs, f, ensure_ascii=False, indent=4)
    print(f"\nAll iteration graphs saved to: {output_path}")
    
    return all_graphs

if __name__ == "__main__":
    concept_combination_x = "A child is drinking wine"
    combination_theme_y = "underage drinking"
    
    all_graphs = generate_and_save_iterative_graphs(concept_combination_x, combination_theme_y)
    combine_list, concept_list = extract_concept_from_graph(all_graphs)
    print(f"combine_list: {combine_list}")
    print(f"concept_list: {concept_list}")
