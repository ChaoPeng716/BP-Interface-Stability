from openai import OpenAI
from pathlib import Path
import os
import re
import json
import glob
import tqdm
from multiprocessing import Pool
from functools import partial
from collections import Counter

API_KEY = ""
BASE_URL = ""
MODEL_GPT = "gpt-4o-mini"

# Ensure the output is a standard JSON format string
def comfirm_json_string_gpt(json_string):
    client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

    prompt = f"""
        You will read a <string>, please fix this string into a string that can be parsed by json.loads.

        Note:
        1. No descriptive text is required.
        2. Don't use markdown syntax.

        The <string>: {json_string}
    """

    response = client.chat.completions.create(
        model=MODEL_GPT,
        messages=[
            {"role": "system", "content": "You are an assistant who is proficient in material synthesis."},
            {"role": "user", "content": prompt}
        ]
    )

    return response.choices[0].message.content

# Ensure the output is a standard JSON format string
def comfirm_json_string(json_string):
    json_string = re.sub(r'[“”]', '"', json_string)
    json_string = re.sub(r'\\', r'\\\\', json_string)
    json_string = re.sub(r'\\"', r'\"', json_string)
    json_string = json_string.replace("\n", "").replace("\r", "")
    # Remove the Markdown syntax wrapping
    if json_string.startswith("```json"):
        json_string = json_string.strip("`json\n")
    json_string = json_string.strip('`\n')

    return json_string

# Text segmentation
def split_by_heading(markdown_text, heading_level='#'):
    # `heading_level` could be '#', '##', '###', etc.
    pattern = r'(?=\n{})'.format(re.escape(heading_level))
    
    # Use regular expression to split, including content with headings
    split_texts = re.split(pattern, markdown_text)
    
    # Remove blank blocks
    return [block.strip() for block in split_texts if block.strip()]

# Text segment classification
def segment_classification(text_split):
    client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

    prompt = f"""
        You will read a text segment about black phosphorus. Please analyze which part of a paper this segment belongs to and give your classification result. The categories you can only choose are as follows:
        1. Abstract
        2. Introduction
        3. Materials and methods
        4. Results and discussion
        5. Conclusions
        6. References

        Please output the result using the following format:
        Category: Abstract/Introduction/Materials and methods/Results and discussion/Conclusions/References 

        Text segment as follows: {text_split}
    """

    response = client.chat.completions.create(
        model=MODEL_GPT,
        messages=[
            {"role": "system", "content": "You are an expert in interdisciplinary research involving materials chemistry, surface and interface science, and the functionalization of nanomaterials."},
            {"role": "user", "content": prompt}
        ]
    )

    return response.choices[0].message.content

# Process a single md file
def process_file(md_path, output_dir):
    chunks = []
    with open(md_path, 'r', encoding='utf-8') as file:
        md_content = file.read()

    # Split the text by heading
    content_splits = split_by_heading(md_content)

    id = 0
    for content_split in content_splits:
        id += 1
        chunk = {}
        result = segment_classification(content_split)
        chunk["id"] = id
        chunk["chunk"] = content_split
        chunk["category"] = result[9:]
        chunks.append(chunk)

    output_path = os.path.join(output_dir, os.path.basename(md_path).replace('.md', '.json'))
    with open(output_path, 'w', encoding='utf-8') as json_file:
        json.dump(chunks, json_file, ensure_ascii=False, indent=4)

# Get the processed md
def chunk_done(json_dir):
    jsons = os.listdir(json_dir)
    json_names = [json_name.replace('.json', '') for json_name in jsons]
    return json_names

# Segment and classify text blocks and save as json
def md_segment():
    md_paths = glob.glob("task2_caj2md/**/*.md", recursive=True)
    print("Number of md files:", len(md_paths))
    # Filter already processed files
    output_dir = "task2-chunks"
    json_names = chunk_done(output_dir)
    md_paths = [md_path for md_path in md_paths if os.path.basename(md_path).replace(".md", "") not in json_names]
    print("Number of filtered md files:", len(md_paths))

    # # Set up a multiprocessing pool
    # pool = Pool(processes=32)

    # process_func = partial(process_file, output_dir=output_dir)

    # # imap_unordered will gradually pass data from md_paths to process_func for parallel processing
    # for _ in tqdm.tqdm(pool.imap_unordered(process_func, md_paths), total=len(md_paths)):
    #     pass

    # pool.close()
    # pool.join()

# Extract synthesis protocol
def get_protocol(text):
    client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

    prompt = f"""
        You will read a text excerpt from an article on the surface modification of black phosphorus. Please extract the specific experimental protocol for surface modification of black phosphorus with functional groups.

        Note:
        1. The information you extract must come from the text excerpt(example not included), and fabrication of information is strictly prohibited.
        2. Don't use markdown syntax.

        Please output the result using the following format:
        {{
            "protocol": ""
        }}

        The text except: {text}
    """

    response = client.chat.completions.create(
        model=MODEL_GPT,
        messages=[
            {"role": "system", "content": "You are an expert in researching surface modification of black phosphorus."},
            {"role": "user", "content": prompt}
        ]
    )

    return  response.choices[0].message.content

# Extract synthesis protocol
def extract_info(chunks_path):
    with open(chunks_path, 'r', encoding='utf-8') as file:
        chunks = json.load(file)

    protocol_dict = {"protocol" : ""}  # Store the final output
    for chunk in chunks:
        chunk_text = chunk['chunk']
        category = chunk['category']
        try:
            # Extract specific experimental steps for molecular modification of black phosphorus surface
            if category == ' Introduction' or category == ' Materials and methods':
                intermediate_result = get_protocol(chunk_text)
                intermediate_result = comfirm_json_string(intermediate_result)
                try:
                    result_protocol = json.loads(intermediate_result)
                except json.JSONDecodeError as e:
                    # Fix JSON string (gpt)
                    escaped_protocol = comfirm_json_string_gpt(intermediate_result)
                    try:
                        result_protocol = json.loads(escaped_protocol)
                    except Exception as e:
                        print(e)
                        print(escaped_protocol)
                        return
                if result_protocol == "":
                    continue
                if isinstance(result_protocol, dict):
                    protocol_dict["protocol"] += result_protocol["protocol"]
                else:
                    print("esult_protocol is not a dictionary")
                    print(result_protocol)
                    return
        except Exception as e:
            print(e)
            return

    output_path = os.path.join(output_dir, os.path.basename(chunks_path))
    with open(output_path, 'w') as json_file:
        json.dump(protocol_dict, json_file, ensure_ascii=False, indent=4)

if __name__ == '__main__':
    chunks_dir = "task2-chunks"
    paths = [os.path.join(chunks_dir, path) for path in os.listdir(chunks_dir)]
    print("Number of chunk files:", len(paths))

    # Filter processed files
    output_dir = "task2-paper-info"
    proccessed_files = [path for path in os.listdir(output_dir)]
    paths = [path for path in paths if os.path.basename(path) not in proccessed_files]
    print("Number of filtered chunks files:", len(paths))

    # # Set up a multiprocessing pool
    # pool = Pool(processes=32)

    # process_func = partial(extract_info)

    # # imap_unordered will gradually pass data from md_paths to process_func for parallel processing
    # for _ in tqdm.tqdm(pool.imap_unordered(process_func, paths), total=len(paths)):
    #     pass

    # pool.close()
    # pool.join()
