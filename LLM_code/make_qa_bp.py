"""
It uses the OpenAI API and MySQL for storing and retrieving data.
@author: Yutang Li
"""
import multiprocessing
import os
import re
import json
import random
import time
import tqdm
import glob
import datetime
import sqlite3
import copy
import multiprocessing
from functools import partial
from openai import OpenAI, APIError  # Ensure correct exception class is imported
from mysql.connector import pooling, Error
from make_qa_prompts_bp import FUNC_GROUPS_QUESTION_RPOMPT, FUNC_GROUPS_ANSWER_PROMPT, PROTOCOL_QUESTION_RPOMPT, PROTOCOL_ANSWER_RPOMPT, SELECT_QUESTION_PROMPT


# Constants
OPENAI_BASE_URL = ""
# OPENAI_BASE_URL = ""
OPENAI_API_KEY = ""
# MODEL_GPT = "gpt-4o-mini"
MODEL_GPT = "chatgpt-4o-latest"
# MINI_MODEL_NAME = "gpt-4o-2024-08-06"
# MAX_MODEL_NAME = "claude-3-5-sonnet-20240620"
# MAX_MODEL_NAME = "gpt-4o-2024-11-20"
# MODEL_GEMINI = "gemini-1.5-flash-002"
MODEL_GEMINI = "gemini-1.5-pro-latest"
MODEL_CLAUDE = "claude-3-5-sonnet-20240620"
# MYSQL_TABLE_NAME = "cif_qa_1104"
cur_dirname = os.path.dirname(__file__)
# DOC_DIR_NAME = os.path.join(cur_dirname, "qa_source_md")
PROCESS = 32  # Number of parallel processes

def record_exists(mp_id, table_name):
    """Check if a mp_id already exists in the table."""
    db = connection_pool.get_connection()
    cursor = db.cursor()
    query = f"SELECT * FROM {table_name} WHERE mp_id = %s"
    cursor.execute(query, (mp_id,))
    result = cursor.fetchone()
    cursor.fetchall()  # Ensure all results are processed
    cursor.close()
    db.close()
    return result is not None

def insert_record(entry, table_name):
    """Insert a record into the MySQL table."""
    db = None
    cursor = None
    try:
        db = connection_pool.get_connection()
        cursor = db.cursor()

        insert_query = f"""
            INSERT INTO {table_name} 
            (mp_id, question_model, question, answer_model, answer, answer_len) 
            VALUES (%s, %s, %s, %s, %s, %s)
        """
        values = (
            entry["mp_id"], entry["question_model"],
            entry["question"], entry["answer_model"], entry["answer"], entry["answer_len"],
        )
        cursor.execute(insert_query, values)
        db.commit()

    except Error as e:
        print(f"Error: {e}")
        db.rollback()
    finally:
        # Ensure cursor is closed
        if cursor:
            cursor.close()
        # Ensure connection is returned to the pool
        if db:
            db.close()

def round_values(data):
    """
    Recursively round all values in the dictionary to three decimal places.
    """
    if isinstance(data, dict):  # if it's a dictionary
        return {key: round_values(value) for key, value in data.items()}
    elif isinstance(data, list):  # If it is a list, recursively process each element
        return [round_values(item) for item in data]
    elif isinstance(data, (int, float)):  # If it is a number, keep three decimal places
        return round(data, 3)
    else:  # For other types, return directly
        return data


def remove_null_values(d):
    """
    Recursively remove key-value pairs with null (None) values from a dictionary.

    Args:
        d (dict): The dictionary to clean.

    Returns:
        dict: A new dictionary without null values.
    """
    if not isinstance(d, dict):
        raise ValueError("Input must be a dictionary")
    _d = copy.deepcopy(d)

    def recursive_remove(d):
        cleaned_dict = {}
        for key, value in d.items():
            if isinstance(value, dict):
                # Recursively clean nested dictionaries
                nested_cleaned = recursive_remove(value)
                if nested_cleaned:  # Only add non-empty dictionaries
                    cleaned_dict[key] = nested_cleaned
            elif value is not None and key != 'version':
                cleaned_dict[key] = value

        return cleaned_dict
    
    clean_dict = recursive_remove(d)
    if _d['cbm'] is None and _d['vbm'] is None:
        # clean_dict['band_gap'] = None
        clean_dict.pop('band_gap')
    return clean_dict

def extract_json(text):
    """Extract JSON content from a block of text using regex."""
    json_pattern = re.compile(r'\\{(?:[^{}]|(?R))*\\}')
    matches = json_pattern.search(text)
    if matches:
        json_str = matches.group(0)
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            return None
    return None

def extract_and_parse_json(response):
    """Extract and parse JSON from a response."""
    json_match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', response)
    json_str = json_match.group(1) if json_match else response.strip()
    json_str = re.sub(r'(\$[^\$]*\$)', lambda m: m.group(1).replace('\\', '\\\\'), json_str)
    json_str = json_str.replace('\\"', '"').replace("\\'", "'")
    try:
        return json.loads(json_str)
    except json.JSONDecodeError as e:
        print(f"JSON parse error: {e}")
        return 'errformat'


# Calculate the tokens of the input message
def count_message_tokens(messages, model_name):
    encoding = tiktoken.encoding_for_model(model_name)
    num_tokens = 0

    num_tokens += len(encoding.encode(messages))
    
    return num_tokens


def generate_func_groups_question(func_groups_info, model_name):
    """Generate a question from the source material using OpenAI with stream."""
    try:
        # Replace context and topic
        instruction = FUNC_GROUPS_QUESTION_RPOMPT.replace("{CONTEXT}", func_groups_info)
        
        # Create an OpenAI client
        client = OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL)

        # Request non-streaming output
        completion = client.chat.completions.create(
            model=model_name,
            stream=False,  # Disable streaming mode
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": instruction}
            ],
        )    

        response = completion.choices[0].message.content

        # Request streaming output
        # completion = client.chat.completions.create(
        #     model=model_name,
        #     stream=True,  # Enable streaming mode
        #     messages=[
        #         {"role": "system", "content": "You are a helpful assistant."},
        #         {"role": "user", "content": instruction}
        #     ],
        # )

        # response = ""  # For accumulating response content
        # # Read and process streaming data step by step
        # for chunk in completion:
        #     if chunk.choices[0].delta.content is not None:
        #         content = chunk.choices[0].delta.content
        #         response += content

        # Parse as JSON response
        json_response = extract_and_parse_json(response)
        if json_response == "errformat":
            return 'errformat'
        return json_response['questions']  # Return the instruction and the parsed response

    except APIError as api_error:
        print(f"generate_design_question API error: {api_error}")
        time.sleep(30)
        return 'apierror'
    except Exception as e:
        print(f"generate_design_question Unexpected error: {e}")
        return 'unexpectederror'


def generate_protocol_question(protocol_info, model_name):
    """Generate a question from the source material using OpenAI with stream."""
    try:
        # Replace context and topic
        instruction = PROTOCOL_QUESTION_RPOMPT.replace("{CONTEXT}", protocol_info)
        
        # Create an OpenAI client
        client = OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL)

        completion = client.chat.completions.create(
            model=model_name,
            stream=False,  # Disable streaming mode
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": instruction}
            ],
        )    

        response = completion.choices[0].message.content

        # # Request streaming output
        # completion = client.chat.completions.create(
        #     model=model_name,
        #     stream=True,  # Enable streaming mode
        #     messages=[
        #         {"role": "system", "content": "You are a helpful assistant."},
        #         {"role": "user", "content": instruction}
        #     ],
        # )

        # response = ""  # For accumulating response content
        # # Read and process streaming data step by step
        # for chunk in completion:
        #     if chunk.choices[0].delta.content is not None:
        #         content = chunk.choices[0].delta.content
        #         response += content

        # Parse into JSON response
        json_response = extract_and_parse_json(response)
        if json_response == "errformat":
            return 'errformat'
        return json_response['questions']  # Return the instruction and the parsed response

    except APIError as api_error:
        print(f"generate_design_question API error: {api_error}")
        return 'apierror'
    except Exception as e:
        print(f"generate_design_question Unexpected error: {e}")
        return 'unexpectederror'


def select_best_question(question_list, answer, model_name):
    try:
        # Replace context and topic
        instruction = SELECT_QUESTION_PROMPT.replace("{ANSWER}", answer).replace("{QUESTIONS}", json.dumps(question_list))
        
        # Create an OpenAI client
        client = OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL)

        # Request non-streaming output
        completion = client.chat.completions.create(
            model=model_name,
            stream=False,  # Disable streaming mode
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": instruction}
            ],
        )    

        response = completion.choices[0].message.content

        # Request streaming output
        # completion = client.chat.completions.create(
        #     model=model_name,
        #     stream=True,  # Enable streaming mode
        #     messages=[
        #         {"role": "system", "content": "You are a helpful assistant."},
        #         {"role": "user", "content": instruction}
        #     ],
        # )

        # response = ""  # For accumulating response content
        # # Read and process streaming data step by step
        # for chunk in completion:
        #     if chunk.choices[0].delta.content is not None:
        #         content = chunk.choices[0].delta.content
        #         response += content

        # Parse into JSON response
        json_response = extract_and_parse_json(response)

        return json_response['questions']  # Return the instruction and the parsed response

    except APIError as api_error:
        print(f"select_best_question API error: {api_error}")
        time.sleep(30)
        return 'apierror'
    except Exception as e:
        print(f"select_best_question Unexpected error: {e}")
        return 'unexpectederror'


def generate_func_groups_answer(question, func_groups_info, sup_content, model_name):
    """Generate an answer to a question using OpenAI with stream."""
    try:
        instruction = FUNC_GROUPS_ANSWER_PROMPT.replace("{QUESTION}", question).replace("{func_groups_info}", func_groups_info).replace("{sup_content}", sup_content)
        client = OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL)

        # Use non-streaming output
        completion = client.chat.completions.create(
            model=model_name,
            stream=False,  # Disable streaming mode
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": instruction}
            ],
        )    

        response = completion.choices[0].message.content

        # Use streaming output
        # completion = client.chat.completions.create(
        #     model=model_name,
        #     stream=True,  # Enable streaming mode
        #     messages=[{"role": "system", "content": "You are a helpful assistant."},
        #               {"role": "user", "content": instruction}],
        # )

        # response = ""  # For accumulating response content
        # # Read and process streaming data step by step
        # for chunk in completion:
        #     if chunk.choices[0].delta.content is not None:
        #         content = chunk.choices[0].delta.content
        #         response += content

        return response.replace("placeholder", "").replace("Placeholder", "")
    
    except APIError as api_error:
        print(f"generate_design_answer API error: {api_error}")
        time.sleep(30)
        return 'apierror'
    except Exception as e:
        print(f"generate_design_answer Unexpected error: {e}")
        return 'unexpectederror'


def generate_protocol_answer(question, protocol_info, model_name):
    """Generate an answer to a question using OpenAI with stream."""
    try:
        instruction = PROTOCOL_ANSWER_RPOMPT.replace("{QUESTION}", question).replace("{context}", protocol_info)
        client = OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL)

        completion = client.chat.completions.create(
            model=model_name,
            stream=False,  # Disable stream mode
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": instruction}
            ],
        )    

        response = completion.choices[0].message.content

        # # Use streaming output
        # completion = client.chat.completions.create(
        #     model=model_name,
        #     stream=True,  # Enable streaming mode
        #     messages=[{"role": "system", "content": "You are a helpful assistant."},
        #               {"role": "user", "content": instruction}],
        # )

        # response = ""  # For accumulating response content
        # # Read and process streaming data step by step
        # for chunk in completion:
        #     if chunk.choices[0].delta.content is not None:
        #         content = chunk.choices[0].delta.content
        #         response += content

        return response.replace("placeholder", "").replace("Placeholder", "")
    
    except APIError as api_error:
        print(f"generate_design_answer API error: {api_error}")
        return 'apierror'
    except Exception as e:
        print(f"generate_design_answer Unexpected error: {e}")
        return 'unexpectederror'


def generate_func_groups_qa(file_content, sup_content, model_name):
    # 1.Generate candidate questions
    func_groups_info = str(file_content)
    # question = generate_question(question_context, MAX_MODEL_NAME)
    question = generate_func_groups_question(func_groups_info, model_name)
    retry = 0
    while (question=='errformat' or question == 'apierror' or question == 'unexpectederror') and retry < 3:
        question = generate_func_groups_question(func_groups_info, model_name)
        retry += 1
    # print(question)
    # 2. Design a scoring system for Q&A pairs to filter out the best question
    # score = select_best_question(question, pre_answer, MAX_MODEL_NAME)
    score = select_best_question(question, func_groups_info, model_name)
    retry = 0
    while (score=='errformat' or score == 'apierror' or score == 'unexpectederror') and retry < 3:
        score = select_best_question(question, func_groups_info, model_name)
        retry += 1
    score = sorted(score, key=lambda x: x['score'], reverse=True)
    q_idx = score[0]['id'] - 1
    # 3. Generate answer
    pre_answer = generate_func_groups_answer(question[q_idx]['text'], func_groups_info, sup_content, model_name)
    retry = 0
    while (pre_answer=='errformat' or pre_answer == 'apierror' or pre_answer == 'unexpectederror') and retry < 3:
        pre_answer = generate_func_groups_answer(question[q_idx]['text'], func_groups_info, sup_content, model_name)
        retry += 1
    return  question[q_idx]['text'], pre_answer


def generate_protocol_qa(protocol_info, model_name):
    # 1.Generate candidate questions
    question = generate_protocol_question(protocol_info, model_name)
    retry = 0
    while (question=='errformat' or question == 'apierror' or question == 'unexpectederror') and retry < 3:
        question = generate_protocol_question(protocol_info, model_name)
        retry += 1
    # print(question)
    # 2. Design a scoring system for Q&A pairs to filter out the best questions
    score = select_best_question(question, protocol_info, model_name)
    retry = 0
    while (score=='errformat' or score == 'apierror' or score == 'unexpectederror') and retry < 3:
        score = select_best_question(question, protocol_info, model_name)
        retry += 1
    score = sorted(score, key=lambda x: x['score'], reverse=True)
    q_idx = score[0]['id'] - 1
    # 3. Generate answer
    pre_answer = generate_protocol_answer(question[q_idx]['text'], protocol_info, model_name)
    retry = 0
    while (pre_answer=='errformat' or pre_answer == 'apierror' or pre_answer == 'unexpectederror') and retry < 3:
        pre_answer = generate_protocol_answer(question[q_idx]['text'], protocol_info, model_name)
        retry += 1
    return  question[q_idx]['text'], pre_answer


# Processing function for a single file
def process_file(input_path, task_id):
    # print(input_path)
    with open(input_path, 'r', encoding='utf-8') as file:
        file_content = json.load(file)
    # print()

    if task_id == "task-1":
        with open("../supplementary/sup_content.md", "r", encoding='utf-8') as file:
            sup_content = file.read()
        # model_list = [MODEL_GPT, MODEL_GEMINI, MODEL_CLAUDE]
        model = MODEL_GPT
        try:
            # Do not process data with empty experimental protocols
            if file_content == {}:
                return
            design_question, design_answer = generate_func_groups_qa(file_content, sup_content, model)

            data = {
                "design_question": design_question,
                "design_answer": design_answer
            }

            output_path = os.path.join(output_task1, os.path.basename(input_path))
            # Write data to a JSON file
            with open(output_path, 'w', encoding='utf-8') as file:
                json.dump(data, file, ensure_ascii=False, indent=2)
            
            # output_path = '../task-1/output.txt'
            # with open(output_path, 'a') as txt_file:
            #     txt_file.write(f"{model} task-1Question:\n")
            #     txt_file.write(design_question+'\n')
            #     txt_file.write(f"{model} task-1Answer:\n")
            #     txt_file.write(design_answer+'\n\n\n')

        except Exception as e:
            print(f"Error processing file: {input_path}")
            print(e)
    elif task_id == "task-2":
        try:
            # Select model
            model = MODEL_GPT
            # model_list = [MODEL_GPT, MODEL_GEMINI, MODEL_CLAUDE]
            protocol_info = file_content["protocol"]
            # Do not process data with empty experimental protocols
            if protocol_info == "":
                return
            design_question, design_answer = generate_protocol_qa(protocol_info, model)

            data = {
                "design_question": design_question,
                "design_answer": design_answer
            }

            output_path = os.path.join(output_task2, os.path.basename(input_path))
            # Write data to a JSON file
            with open(output_path, 'w', encoding='utf-8') as file:
                json.dump(data, file, ensure_ascii=False, indent=2)
            
            # output_path = os.path.join(task2_dir, os.path.basename(input_path).replace('.json', '.txt'))
            # with open(output_path, 'a') as txt_file:
            #     txt_file.write(f"{model} task-2Question:\n")
            #     txt_file.write(design_question+'\n')
            #     txt_file.write(f"{model} task-2Answer:\n")
            #     txt_file.write(design_answer+'\n\n\n')

        except Exception as e:
            print(f"Error processing file: {input_path}")
            print(e)


if __name__ == "__main__":
    task1_dir = "../task-1"
    task2_dir = "../task-2"

    # The path to the information extracted in Task 1
    task1_paper_info_jsons_dir = os.path.join(cur_dirname, task1_dir, "task1-paper-info")
    task1_paper_info_jsons_paths = [os.path.join(task1_paper_info_jsons_dir, path) for path in os.listdir(task1_paper_info_jsons_dir)]
    print("Total number of task1 files", len(task1_paper_info_jsons_paths))
    # Filter processed files
    output_task1 = os.path.join(task1_dir, "task1-qa-new")
    processed_task1 = [path for path in os.listdir(output_task1)]
    task1_paper_info_jsons_paths = [path for path in task1_paper_info_jsons_paths if os.path.basename(path) not in processed_task1]
    print("Number of files in task1 after filtering:", len(task1_paper_info_jsons_paths))

    # Get the path of the information extracted in Task 2
    task2_paper_info_jsons_dir = os.path.join(cur_dirname, task2_dir, "task2-paper-info")
    task2_paper_info_jsons_paths = [os.path.join(task2_paper_info_jsons_dir, path) for path in os.listdir(task2_paper_info_jsons_dir)]
    print("Total number of task2 files:", len(task2_paper_info_jsons_paths))
    # Filter processed files
    output_task2 = os.path.join(task2_dir, "task2-qa")
    processed_task2 = [path for path in os.listdir(output_task2)]
    task2_paper_info_jsons_paths = [path for path in task2_paper_info_jsons_paths if os.path.basename(path) not in processed_task2]
    print("Number of files in task2 after filtering:", len(task2_paper_info_jsons_paths))

    # Switch between task-1 and task-2
    process_file_with_params = partial(process_file, task_id = "task-1")

    # task-1
    # with multiprocessing.Pool(32) as pool:
    #     # Use tqdm to track progress
    #     for _ in tqdm.tqdm(pool.imap_unordered(process_file_with_params, task1_paper_info_jsons_paths), total=len(task1_paper_info_jsons_paths)):
    #         pass
    
    # task-2
    # with multiprocessing.Pool(16) as pool:
    #     # Use tqdm to track progress
    #     for _ in tqdm.tqdm(pool.imap_unordered(process_file_with_params, task2_paper_info_jsons_paths), total=len(task2_paper_info_jsons_paths)):
    #         pass
