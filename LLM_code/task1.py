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
def comfirm_json_string(response):
    """Extract and parse JSON from a response."""
    json_match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', response)
    json_str = json_match.group(1) if json_match else response.strip()
    json_str = re.sub(r'(\$[^\$]*\$)', lambda m: m.group(1).replace('\\', '\\\\'), json_str)
    json_str = json_str.replace('\\"', '"').replace("\\'", "'")
    
    return json_str
    
# Text segmentation
def split_by_heading(markdown_text, heading_level='#'):
    # `heading_level` could be '#', '##', '###', etc.
    pattern = r'(?=\n{})'.format(re.escape(heading_level))
    
    # Split using regular expressions to include the content with titles
    split_texts = re.split(pattern, markdown_text)
    
    # Remove blank blocks
    return [block.strip() for block in split_texts if block.strip()]

# Text segment classification
def segment_classification(text_split):
    client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

    prompt = f"""
        You will read a text segment about hydrophilic polymers. Please analyze which part of a paper this segment belongs to and give your classification result. The categories you can only choose are as follows:
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
            {"role": "system", "content": "You are an expert in interdisciplinary research across fields such as materials chemistry, polymer science, biomaterials engineering, and interface and surface science."},
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
    md_paths = glob.glob("task1_caj2md/mds/**/*.md", recursive=True)
    print("Number of md files:", len(md_paths))
    # Filter already processed files
    output_dir = "task1-chunks"
    json_names = chunk_done(output_dir)
    md_paths = [md_path for md_path in md_paths if os.path.basename(md_path).replace(".md", "") not in json_names]
    print("Number of filtered md files:", len(md_paths))
    
    for path in tqdm.tqdm(md_paths):
        try:
            process_file(path, output_dir)
        except Exception as e:
            print(f"Error occurred while processing {path}: {e}")

    # # Set up a multiprocessing pool
    # pool = Pool(processes=32)

    # process_func = partial(process_file, output_dir=output_dir)

    # # imap_unordered will gradually pass data from md_paths to process_func for parallel processing
    # for _ in tqdm.tqdm(pool.imap_unordered(process_func, md_paths), total=len(md_paths)):
    #     pass

    # pool.close()
    # pool.join()

# Extract the corresponding functional groups for surface modification to enhance the stability of black phosphorus
def get_function_groups(text):
    client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

    prompt = f"""
        You are an expert in the field of surface modification of black phosphorus. You will read a text excerpt from an article on the surface modification of black phosphorus to improve the surface stability of black phosphorus. Please extract the important information from it, including (1) what groups of molecules can be used for surface modification to stabilize black phosphorus, and (2) how molecules containing these groups are used for surface modification to stabilize black phosphorus.

        Note:
        1. The information you extract must come from the text excerpt(example not included), and fabrication of information is strictly prohibited.
        2. Don't use markdown syntax.

        Please output the result using the following format:
        {{
            "groups of molecules": "how molecules are used for surface modification",
            ...,
            "groups of molecules": "how molecules are used for surface modification"
        }}

        The text except: {text}
    """

    response = client.chat.completions.create(
        model=MODEL_GPT,
        messages=[
             {"role": "system", "content": "You are an expert in the field of surface modification of black phosphorus."},
            {"role": "user", "content": prompt}
        ]
    )

    return  response.choices[0].message.content

# Extract functional group information
def extract_info(chunks_path):
    with open(chunks_path, 'r', encoding='utf-8') as file:
        chunks = json.load(file)

    function_group_dict = {}   # store the final output
    for chunk in chunks:
        chunk_text = chunk['chunk']
        category = chunk['category']
        try:
            # Extract the corresponding functional groups for surface modification to enhance the stability of black phosphorus
            if category == ' Abstract' or category == ' Introduction' or category == ' Materials and methods' or category == 'Results and discussion' or category == ' Conclusions':
                intermediate_result = get_function_groups(chunk_text)
                print(intermediate_result)
                intermediate_result = comfirm_json_string(intermediate_result)
                try:
                    result_function_group = json.loads(intermediate_result)
                except json.JSONDecodeError as e:
                    # Fix JSON string (gpt)
                    escaped_function_group = comfirm_json_string_gpt(intermediate_result)
                    try:
                        result_function_group = json.loads(escaped_function_group)
                    except Exception as e:
                        print(f"JSON parsing failed: {e}")
                        print(f"Original output: {escaped_function_group}")
                        return
                
                # Check if the returned dictionary is empty or contains no valid content
                if not isinstance(result_function_group, dict):
                    print("result_function_group is not a dictionary")
                    print(result_function_group)
                    return
                else:
                    function_group_dict.update(result_function_group)
                    
        except Exception as e:
            print(f"Error occurred while processing chunk: {e}")
            return

    output_path = os.path.join(output_dir, os.path.basename(chunks_path))
    with open(output_path, 'w', encoding='utf-8') as json_file:
        json.dump(function_group_dict, json_file, ensure_ascii=False, indent=4)

if __name__ == '__main__':
    chunks_dir = "task1-chunks"
    paths = [os.path.join(chunks_dir, path) for path in os.listdir(chunks_dir)]
    print("Number of chunk files:", len(paths))

    # Filter processed files
    output_dir = "task1-paper-info"
    proccessed_files = [path for path in os.listdir(output_dir)]
    paths = [path for path in paths if os.path.basename(path) not in proccessed_files]
    print("Number of filtered chunks files:", len(paths))
    
    # step1
    # md_segment()
    
    # step2
    for path in tqdm.tqdm(paths[:2]):
        try:
            extract_info(path)
        except Exception as e:
            print(f"Error processing {path}: {e}")
    
    
    # Set up a multiprocessing pool
    # pool = Pool(processes=32)

    # process_func = partial(extract_info)

    # # imap_unordered will gradually pass data from md_paths to process_func for parallel processing
    # for _ in tqdm.tqdm(pool.imap_unordered(process_func, paths), total=len(paths)):
    #     pass

    # pool.close()
    # pool.join()
