"""
    StorytellingPainter Pipeline.
"""


import os
import json
import base64
import argparse
import requests
import pandas as pd
from openai import OpenAI
from tqdm import tqdm
from utils import read_jsonl


class StorytellingPainter:
    def __init__(self, llm_name=None, story_gen_mode=None, img_gen_model_name=None, 
                 N=None, output_dir=None, story_pool_name=None, llm_temperature=1.0):
        self.llm_name = llm_name
        self.story_gen_mode = story_gen_mode
        self.img_gen_model_name = img_gen_model_name
        self.N = N
        self.llm_temperature = llm_temperature
        self.output_dir = output_dir
        self.story_output_file = os.path.join(output_dir, "story_output_{}.jsonl".format(self.llm_name))
        self.img_gen_output_file = os.path.join(output_dir, "img_output_{}_{}.jsonl".format(self.llm_name, self.img_gen_model_name))
        self.image_dir = os.path.join(output_dir, "image_{}_{}".format(self.llm_name, self.img_gen_model_name))
        self.api_key = "your api key here."
        self.model_config = {"name": self.llm_name, "key": self.api_key}
        
        if story_pool_name is not None:
            if story_pool_name == "cogbench":
                self.story_pool = pd.read_csv("system/cogbench_story_pool.csv")
            elif story_pool_name == "cogbench_train":
                self.story_pool = pd.read_csv("system/cogbench_story_pool_train.csv")
            elif story_pool_name == "cogbench_test":
                self.story_pool = pd.read_csv("system/cogbench_story_pool_test.csv")
            else:
                raise ValueError("Invalid story pool name.")
        

    def chat_gpt_gen(self, system_prompt, user_input):  
        """
            Call ChatGPT.
        """
        client = OpenAI(api_key = self.model_config['key'],
                        base_url = "base url here.") 
        
        if system_prompt != "":
            response = client.chat.completions.create(
                model=self.model_config['name'],
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_input}
                ],
                max_tokens=1024,
                temperature=self.llm_temperature
            )
        else:
            response = client.chat.completions.create(
                model=self.model_config['name'],
                messages=[
                    {"role": "user", "content": user_input}
                ],
                max_tokens=1024,
                temperature=self.llm_temperature
            )
        
        return response.choices[0].message.content.strip()


    ####### Story Generation #######
    def storytelling_single(self):
        """
            Generate 1 story using ChatGPT.
        """
        
        if self.story_gen_mode == "naive":
            system_prompt = open("system/naive_story_gen_system.txt", encoding="utf-8").read()
        elif self.story_gen_mode == "cor_guided":
            system_prompt = open("system/cor_story_gen_system.txt", encoding="utf-8").read()
        else:
            raise ValueError("Invalid mode.")

        if self.story_gen_mode=="naive":
            example_ids = []
            examples = []
            user_input = "Please imagine a storytelling image and describe in detail the content and the story that unfolds within it."
        elif self.story_gen_mode=="cor_guided":
            sampled_story_df = self.story_pool.sample(3)
            example_ids = list(sampled_story_df['img_id'])
            examples = list(sampled_story_df['story'])
            user_input = "Here are some examples of stories depicted in images:\n\nExample 1:\n{}\n\nExample 2:\n{}\n\nExample 3:\n{}\n\nPlease imagine a storytelling image and describe in detail the content and the story that unfolds within it.".format(examples[0], examples[1], examples[2])
        else:
            raise ValueError("Invalid mode.")

        model_output = self.chat_gpt_gen(system_prompt, user_input)

        return example_ids, examples, model_output


    def storytelling_batch(self):
        """
            Generate N stories using ChatGPT and save to file.
        """

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        if self.story_gen_mode=="naive":
            system_prompt = open("system/naive_story_gen_system.txt", encoding="utf-8").read()
        elif self.story_gen_mode=="cor_guided":
            system_prompt = open("system/cor_story_gen_system.txt", encoding="utf-8").read()
        else:
            raise ValueError("Invalid mode")

        response = {}
        for i in tqdm(range(self.N)):

            if self.story_gen_mode=="naive":
                user_input = "Please imagine a storytelling image and describe in detail the content and the story that unfolds within it."
            elif self.story_gen_mode=="cor_guided":
                sampled_story_df = self.story_pool.sample(3)
                example_ids = list(sampled_story_df['img_id'])
                examples = list(sampled_story_df['story'])
                user_input = "Here are some examples of stories depicted in images:\n\nExample 1:\n{}\n\nExample 2:\n{}\n\nExample 3:\n{}\n\nPlease imagine a storytelling image and describe in detail the content and the story that unfolds within it.".format(examples[0], examples[1], examples[2])
            else:
                raise ValueError("Invalid mode")

            model_output = self.chat_gpt_gen(system_prompt, user_input)

            print("Model Output: ", model_output)
            
            if self.story_gen_mode=="cor_guided":
                response["example1_id"] = example_ids[0]
                response["example2_id"] = example_ids[1]
                response["example3_id"] = example_ids[2]

                response["example1"] = examples[0]
                response["example2"] = examples[1]
                response["example3"] = examples[2]

            response["model_output"] = model_output

            with open(self.story_output_file, "a") as wf:
                json.dump(response, wf)
                wf.write('\n')


    ####### Image Generation #######
    ### models    
    def gpt_img_1_img_gen(self, description, file_name):
        """
            GPT-image-1 text 2 image API.  
        """

        try:
            client = OpenAI(api_key = self.model_config['key'],
                            base_url = "") 

            result = client.images.generate(
                model="gpt-image-1",
                prompt=description,
                # quality="medium", #  default is "auto" 
            )

            image_base64 = result.data[0].b64_json
            image_bytes = base64.b64decode(image_base64)

            # Save the image to a file
            image_path = os.path.join(self.image_dir, file_name)
            with open(image_path, "wb") as f:
                f.write(image_bytes)

        except Exception as e:
            print(f"GPT-image-1 error: {str(e)}")
            return

    def dalle3_img_gen(self, description, file_name):

        client = OpenAI(api_key = self.model_config['key'],
                        base_url = "") 

        try:
            response = client.images.generate(
                model="dall-e-3",
                prompt=description,
                size="1024x1024",
                quality="standard",
                n=1,
                response_format="url"  
            )
        except Exception as e:
            print("Error: ", e)
            return

        revised_prompt = response.data[0].revised_prompt

        # save prompts
        img_gen_output = {
            "file_name": file_name,
            "original_prompt": description,
            "revised_prompt": revised_prompt,
            "image_url": response.data[0].url
        }
        with open(self.img_gen_output_file, "a") as wf:
            json.dump(img_gen_output, wf)
            wf.write('\n')

        # save img
        image_url = response.data[0].url  
        image_response = requests.get(image_url)
        image_path = os.path.join(self.image_dir, file_name)
        with open(image_path, "wb") as file:
            file.write(image_response.content)
        

    def story_img_gen(self):
        """
        Generate N stories and corresponding images.
        """
        # create output dir if not exist
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        
        if not os.path.exists(self.image_dir):
            os.makedirs(self.image_dir)

        response = {}
        for i in range(self.N):

            # generate story
            example_ids, examples, generated_story = self.storytelling_single()
            
            print("Model Output: ", generated_story)

            # generate image
            img_file_name = "{}".format(i+1) + ".png"  
            if self.img_gen_model_name == "gpt-image-1": 
                self.gpt_img_1_img_gen(generated_story, img_file_name)
            elif self.img_gen_model_name == "dalle3":
                self.dalle3_img_gen(generated_story, img_file_name)
            else:
                raise ValueError("Invalid image generation model name")

            # saving story
            if self.story_gen_mode=="cor_guided":
                response["example1_id"] = example_ids[0]
                response["example2_id"] = example_ids[1]
                response["example3_id"] = example_ids[2]

                response["example1"] = examples[0]
                response["example2"] = examples[1]
                response["example3"] = examples[2]

            response["model_output"] = generated_story
            response["img_file_name"] = img_file_name

            with open(self.story_output_file, "a") as wf:
                json.dump(response, wf)
                wf.write('\n')


    def story_img_gen_based_on_story_file(self, story_file):
        """
        Generate images based on story files.
        """
        # create output dir if not exist
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        
        if not os.path.exists(self.image_dir):
            os.makedirs(self.image_dir)

        stories = read_jsonl(story_file)

        for i, story_sample in enumerate(tqdm(stories)):
            
            generated_story = story_sample['model_output']
            print(generated_story)
        
            # generate image
            img_file_name = "{}".format(i+1) + ".png"  
            if self.img_gen_model_name == "gpt-image-1":
                self.gpt_img_1_img_gen(generated_story, img_file_name)
            elif self.img_gen_model_name == "dalle3":
                self.dalle3_img_gen(generated_story, img_file_name)
            else:
                raise ValueError("Invalid image generation model name")


def main(args):

    painter = StorytellingPainter(llm_name=args.llm_name, 
                                  story_gen_mode=args.story_gen_mode, 
                                  img_gen_model_name=args.img_gen_model_name, 
                                  llm_temperature=args.llm_temperature,
                                  N=args.N, 
                                  output_dir=args.output_dir,
                                  story_pool_name=args.story_pool_name)
    
    if args.function == "story_gen":
        painter.storytelling_batch()
    elif args.function == "story_img_gen":
        painter.story_img_gen()
    elif args.function == "story_img_gen_based_on_story_file":
        painter.story_img_gen_based_on_story_file(args.story_file)
    else:
        raise ValueError("Invalid function name")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--llm_name", type=str, default=None)
    parser.add_argument("--story_gen_mode", type=str, default=None)
    parser.add_argument("--img_gen_model_name", type=str, default=None)
    parser.add_argument("--llm_temperature", type=float, default=1.0)
    parser.add_argument("--N", type=int, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--function", type=str, default=None)
    parser.add_argument("--story_file", type=str, default=None)
    parser.add_argument("--story_pool_name", type=str, default=None)
    args = parser.parse_args()

    main(args)
