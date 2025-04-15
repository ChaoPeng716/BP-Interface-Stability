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

API_KEY = "sk-sVQdMEtwBB4TuRsg6bF0E0DeCb164cEdA1AbBaBbB390D1C8"
BASE_URL = "https://vip.apiyi.com/v1"
MODEL_GPT = "gpt-4o-mini"

# 确保输出为标准json格式字符串
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

# 确保输出为标准json格式字符串
def comfirm_json_string(json_string):
    json_string = re.sub(r'[“”]', '"', json_string)
    json_string = re.sub(r'\\', r'\\\\', json_string)
    json_string = re.sub(r'\\"', r'\"', json_string)
    json_string = json_string.replace("\n", "").replace("\r", "")
    # 去掉 Markdown 的语法包裹
    if json_string.startswith("```json"):
        json_string = json_string.strip("`json\n")
    json_string = json_string.strip('`\n')

    return json_string

# 文本分割
def split_by_heading(markdown_text, heading_level='#'):
    # `heading_level` could be '#', '##', '###', etc.
    pattern = r'(?=\n{})'.format(re.escape(heading_level))
    
    # 使用正则表达式进行切割，以包含标题的内容
    split_texts = re.split(pattern, markdown_text)
    
    # 去除空白的块
    return [block.strip() for block in split_texts if block.strip()]

# 文本段分类
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

# 处理单个md文件
def process_file(md_path, output_dir):
    chunks = []
    with open(md_path, 'r', encoding='utf-8') as file:
        md_content = file.read()

    # 将文本按heading分割
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

# 获取已经处理过的md
def chunk_done(json_dir):
    jsons = os.listdir(json_dir)
    json_names = [json_name.replace('.json', '') for json_name in jsons]
    return json_names

# 将文本段分割分类并保存为json
def md_segment():
    md_paths = glob.glob("task2_caj2md/**/*.md", recursive=True)
    print("md文件数量：", len(md_paths))
    # 过滤已经过处理的文件
    output_dir = "task2-chunks"
    json_names = chunk_done(output_dir)
    md_paths = [md_path for md_path in md_paths if os.path.basename(md_path).replace(".md", "") not in json_names]
    print("过滤后md文件数量：", len(md_paths))

    # # 设置多进程池
    # pool = Pool(processes=32)

    # process_func = partial(process_file, output_dir=output_dir)

    # # imap_unordered 将逐步从 md_paths 传给 process_func 进行并行处理
    # for _ in tqdm.tqdm(pool.imap_unordered(process_func, md_paths), total=len(md_paths)):
    #     pass

    # pool.close()
    # pool.join()

# 提取合成方案
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

# 提取实验方案
def extract_info(chunks_path):
    with open(chunks_path, 'r', encoding='utf-8') as file:
        chunks = json.load(file)

    protocol_dict = {"protocol" : ""}  # 存放最终输出
    for chunk in chunks:
        chunk_text = chunk['chunk']
        category = chunk['category']
        try:
            # 提取分子做黑磷表面修饰的具体实验步骤
            if category == ' Introduction' or category == ' Materials and methods':
                intermediate_result = get_protocol(chunk_text)
                intermediate_result = comfirm_json_string(intermediate_result)
                try:
                    result_protocol = json.loads(intermediate_result)
                except json.JSONDecodeError as e:
                    # 修复json字符串(gpt)
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
                    print("result_protocol不是一个字典")
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
    print("chunks文件数量：", len(paths))

    # 过滤已处理的文件
    output_dir = "task2-paper-info"
    proccessed_files = [path for path in os.listdir(output_dir)]
    paths = [path for path in paths if os.path.basename(path) not in proccessed_files]
    print("过滤后chunks文件数量：", len(paths))

    # # 设置多进程池
    # pool = Pool(processes=32)

    # process_func = partial(extract_info)

    # # imap_unordered 将逐步从 md_paths 传给 process_func 进行并行处理
    # for _ in tqdm.tqdm(pool.imap_unordered(process_func, paths), total=len(paths)):
    #     pass

    # pool.close()
    # pool.join()
