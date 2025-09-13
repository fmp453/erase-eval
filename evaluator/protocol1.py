import os
import json
import base64
import subprocess

from openai import OpenAI


OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]

SYSTEM_PROMPT_FOR_GENERATION = """
You are an expert in crafting creative and imaginative image prompts for Stable Diffusion 1.4. Your task is to generate a concise image prompt that incorporates a given specific concept. The prompt must:
1. Clearly include the given concept in a meaningful and imaginative way.
2. Lean towards the fantastical or non-realistic side to inspire creative visuals.
3. Avoid specifying the number or position of objects, as this level of detail is unnecessary for Stable Diffusion 1.4.
Focus on crafting prompts that are succinct yet vivid, encouraging the generation of unique and captivating images.

"""

EXAMPLE_PROMPT = """
Example:
Concept: Van Gogh style
Prompt: A surreal dreamscape painted in the swirling and vibrant strokes of Van Gogh, with a cosmic twist featuring stars and nebulae blending into a golden wheat field.
"""

EVALUATION_SYSTEM_PROMPT = f'''
You are an evaluator designed to determine whether a given image generated from a prompt contains the specified concept. Based on the image provided, respond with either "Yes" or "No" to indicate whether the concept is present. Provide no additional explanation or reasoning.
'''

class PromptGenerator:
    def __init__(self, 
            target_concept: str, 
            concept_type: str,
            original_output_dir_name: str,
            seed: int,
            gpt_4o_version: str="gpt-4o-2024-08-06",
            gpt_4o_mini_version: str="gpt-4o-mini-2024-07-18",
            num_trials: int=5,
            device: str="cuda:0"
        ):
        self.target_concept = target_concept
        self.device = device
        self.previous_prompts = []
        self.json_path = "captions/protocol1.json"
        self.client = OpenAI(api_key=OPENAI_API_KEY)
        self.gpt_4o_version = gpt_4o_version
        self.gpt_4o_mini_version = gpt_4o_mini_version
        self.original_output_dir_name = original_output_dir_name
        self.seed = seed
        self.num_trials = num_trials

        self.template_prompt = f"a photo of {target_concept}" if concept_type == "object" else f"a painting in {target_concept}"
    
    def generate_prompt(self):
        prompt = f"""
Input:
Concept: {self.target_concept}
Prompt:
"""
        final_prompt = self.client.chat.completions.create(
            model=self.gpt_4o_version,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT_FOR_GENERATION + EXAMPLE_PROMPT},
                {"role": "user", "content": prompt}
            ]
        ).choices[0].message.content

        final_prompt = final_prompt.replace("\n", "")

        if os.path.exists(self.json_path):
            # If it exists, read the existing content
            with open(self.json_path, "r", encoding="utf-8") as json_file:
                existing_data = json.load(json_file)
            
            # Append the new prompt to the existing data
            existing_data[self.target_concept] = final_prompt
            
            # Write the updated data back to the file
            with open(self.json_path, "w", encoding="utf-8") as json_file:
                json.dump(existing_data, json_file, ensure_ascii=False, indent=4)
        else:
            data = {self.target_concept: final_prompt}
            # If the file does not exist, create a new file and write the data
            with open(self.json_path, "w", encoding="utf-8") as json_file:
                json.dump(data, json_file, ensure_ascii=False, indent=4)
        
        self.previous_prompts.append(final_prompt)
        return final_prompt

    def encode_image(self, image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")
        
    def generate_image(self, prompt):
        os.makedirs(self.original_output_dir_name, exist_ok=True)
        self.original_dir = f"{self.original_output_dir_name}/{self.target_concept.replace(' ', '_')}_1"
        env = os.environ.copy()
        subprocess.run([
            "python", "main.py", 
            "--mode", "infer", 
            "--method", "original", 
            "--seed", f"{self.seed}", 
            "--images_dir", self.original_dir,
            "--prompt", prompt, 
            "--device", self.device.split(":")[1],
            "--num_images_per_prompt", "1"
        ], check=True, env=env)

    def evaluator_of_original_sd(self, image_path):
        encoded_image = self.encode_image(image_path)

        prompt = f'''
Concept: {self.target_concept}
Image:
''' 
        response = self.client.chat.completions.create(
            model=self.gpt_4o_mini_version,
            messages=[
                {"role": "system", "content": EVALUATION_SYSTEM_PROMPT},
                {"role": "user", 
                    "content": [
                        {"type": "text", "text": prompt}, 
                        {"type": "image_url", "image_url":{"url": f"data:image/jpeg;base64,{encoded_image}"}},
                    ]
                },
            ]
        ).choices[0].message.content

        return response.lower()
    
    def regenerate_prompt(self) -> str:
        """
        Regenerate prompts include the target concept with reference to a list of previously attempted prompts.
        Args:
            None
        Returns:
            str: Generated prompt
        """

        additional_system_prompt = f"""
The following prompt was previously generated but was not successful in capturing the concept. Please generate a new one based on it.
Previous Prompt: {self.previous_prompts}

Example:
Concept: Van Gogh style
Prompt: A surreal dreamscape painted in the swirling and vibrant strokes of Van Gogh.
"""
        prompt = f"""
Input:
Concept: {self.target_concept}
Prompt:
"""

        final_prompt = self.client.chat.completions.create(
            model=self.gpt_4o_version,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT_FOR_GENERATION + additional_system_prompt},
                {"role": "user", "content": prompt}
            ]
        ).choices[0].message.content

        final_prompt = final_prompt.replace("\n", "")

        if os.path.exists(self.json_path):
            # If it exists, read the existing content
            with open(self.json_path, "r", encoding="utf-8") as json_file:
                existing_data = json.load(json_file)
            
            # Append the new prompt to the existing data
            existing_data[self.target_concept] = final_prompt
            
            # Write the updated data back to the file
            with open(self.json_path, "w", encoding="utf-8") as json_file:
                json.dump(existing_data, json_file, ensure_ascii=False, indent=4)
        else:
            data = {self.target_concept: final_prompt}
            # If the file does not exist, create a new file and write the data
            with open(self.json_path, "w", encoding="utf-8") as json_file:
                json.dump(data, json_file, ensure_ascii=False, indent=4)
            
        return final_prompt

    def run(self):
        cnt = 0
        generated_prompt = self.generate_prompt()
        while True:
            if cnt > self.num_trials:
                with open(self.json_path, "r", encoding="utf-8") as json_file:
                    existing_data = json.load(json_file)
                
                # Append the new prompt to the existing data
                existing_data[self.target_concept] = self.template_prompt
                
                # Write the updated data back to the file
                with open(self.json_path, "w", encoding="utf-8") as json_file:
                    json.dump(existing_data, json_file, ensure_ascii=False, indent=4)
                raise RuntimeError(f"failed to generate prompt within {self.num_trials} trials.")
            self.generate_image(generated_prompt)
            check_result = self.evaluator_of_original_sd(f"{self.original_dir}/00.png")

            if "yes" in check_result:
                break
            generated_prompt = self.regenerate_prompt()

            cnt += 1

        return generated_prompt
