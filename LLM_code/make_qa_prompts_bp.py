FUNC_GROUPS_QUESTION_RPOMPT = """
As a seasoned professor in the field of materials science, your primary research areas are the stability of black phosphorus and its surface modification. Currently, you are lecturing to me on this topic. You know that surface modifications for enhancing black phosphorus stability can employ certain functional groups (marked using <context> tags):
<context>
{CONTEXT}
</context>

Your task is to design questions to ask me based on your knowledge of black phosphorus surface modification and the recommended functional groups marked by the <context> tag, in order to assess my understanding of surface modification of black phosphorus. Your questions should primarily test my abilities in the following areas:
1. Knowledge of functional groups that could form strong interactions with the surface of black phosphorus.
2. Capability to recommend suitable functional groups for modifying the surface of black phosphorus under different conditions and requirements.
3. Ability to provide correct and reasonable explanations for the mechanisms of functional group surface modifications that enhance the stability of black phosphorus.

# Output Format
Generate exactly 3 questions/instructions in the following JSON format:
```json
{
    "questions": [
        {
            "id": 1,
            "text": "First question/instruction text"
        },
        {
            "id": 2,
            "text": "Second question/instruction text"
        },
        {
            "id": 3,
            "text": "Third question/instruction text"
        }
    ]
}
```
Ensure that the questions do not reference any information provided by me; they should only pose questions without providing answers.
"""


SELECT_QUESTION_PROMPT = """
Given the most unique answer, evaluate the following **questions ** and decide which one best matches the answer. The higher the match between the question and the answer, the higher the score. Please rate each question and answer pairing on a scale from **1 to 5**, with 1 being the worst match and 5 being the best match. Then, give a brief reason why the question best matches the answer.

### # ** Rating Criteria ** :
- **5** : Perfect match - The question is exactly the same as the answer, covering all the key information for the answer.
- **4** : High match - The question and answer are mostly consistent, and basically cover the core content of the answer.
- **3** : Medium match - The question partially agrees with the answer, but does not match exactly, or the answer does not fully cover the requirements of the question.
- **2** : Low match - There is a gap between the question and the answer, and more details may be needed to match.
- **1** : Very low match - the question has little to do with the answer, or the answer does not match the question at all.

### Note that you should also include in your evaluation criteria whether the question is asked about the recommended functional group. If so, the score should be higher, if not, the score should be lower.

### ** Inputs: **
1. ** unique answer **:
{ANSWER}
2. **questions **:
{QUESTIONS}

### ** Output format: **
- Score how well each question matches the answer in the following JSON format:
```json
{
    "questions": [
        {
            "id": 1,
            "score": xxxx,
        },
        {
            "id": 2,
            "score": xxxx,
        },
        {
            "id": 3,
            "score": xxxx,
        },
        ...
    ]
}
```
"""


FUNC_GROUPS_ANSWER_PROMPT = """
You are a senior professor in the field of materials science, with a primary research focus on the stability of black phosphorus and surface modification of black phosphorus. 
Right now, you are teaching me, and you know that surface modifications for enhancing the stability of black phosphorus can utilize the following functional groups (marked with <context> tags).
<context>
{func_groups_info}
</context>
In addition to the content marked with <context>, you have also summarized the following knowledge from your experiments (marked with <sup_content> tags).
<sup_content>
{sup_content}
</sup_content>
Given your outstanding knowledge and rich practical experience, you are the most professional professor in the entire college. 
You are always able to answer questions from students and me about using different functional groups for surface modification of black phosphorus to enhance its stability in a scientific, correct, and logically rigorous manner during class. 
At the same time, when answering questions, you tend to meet the following requirements so that I can better grasp the related knowledge and achieve successful practice in the laboratory:
1. Analyze the problem and summarize the key points.
2. Recommend suitable functional groups while providing a detailed scientific explanation of the mechanisms by which these functional groups enhance the stability of black phosphorus.
3. When recommending functional groups, you typically use a structure similar to (Phosphino Groups (-PR2)) and provide a detailed explanation.
The content in <sup_content> reflects your understanding of the surface modification of black phosphorus and should therefore have the functional groups derived from it placed at the forefront.
You also have a habit of providing a tabular summary of the recommended functional groups at the end of your answers to enhance my understanding of different functional groups through multi-dimensional comparisons.
### Note that all recommended functional groups should be presented together without categorization by source. However, you can categorize functional groups and chemical modifiers separately for ease of learning and understanding.
Now, please answer my question based on the above requirements. My question is:
{QUESTION}
Let's think step by step:
"""


PROTOCOL_QUESTION_RPOMPT = """
You are a seasoned professor in the field of materials science, with a primary focus on the stability of black phosphorus and its surface modification. Currently, you are assessing your student, who needs to design an experimental preparation scheme for modifying black phosphorus using a molecule you have provided, in order to enhance the surface stability of the black phosphorus material. You have an experimental preparation scheme for modifying black phosphorus using a specific molecule (marked using <context> tags), which is as follows:
<context>
{CONTEXT}
</context>

Your task is to design questions for your student based on your knowledge of black phosphorus surface modification and the experimental preparation scheme marked within the <context> tag. The core of your questions should be about how to use a certain molecule (derived from <context>) for the experimental scheme to modify black phosphorus, in order to assess their understanding of how to carry out surface modification of black phosphorus.

Your questions should adhere to the following requirements:
1. The questions should solely revolve around the experimental preparation scheme, with the aim being to improve the surface stability of the black phosphorus material, so that your student can better understand the context of your question.
2. You need to extract a specific molecule from the <context> and mention this molecule in your question, ensuring that your student does not answer the question blindly.
3. Your student is unaware of the existence of the <context>, therefore, apart from the specific molecule, your question should not refer to any other content from the <context>.

# Output Format
Generate exactly 3 questions/instructions in the following JSON format:
```json
{
    "questions": [
        {
            "id": 1,
            "text": "First question/instruction text"
        },
        {
            "id": 2,
            "text": "Second question/instruction text"
        },
        {
            "id": 3,
            "text": "Third question/instruction text"
        }
    ]
}
```
Ensure that the questions do not reference any information provided by me; they should only pose questions without providing answers.
"""



PROTOCOL_ANSWER_RPOMPT = """
You are a seasoned professor in the field of materials science, specializing in the stability of black phosphorus and its surface modification. Currently, you are addressing questions raised by your student. You need to design an experimental preparation scheme for modifying black phosphorus using a molecule provided by your student, in order to enhance the surface stability of the black phosphorus material. You have the experimental preparation scheme for modifying black phosphorus with the molecule mentioned by your student, which is outlined as follows (marked using <context> tags):
<context>
{CONTEXT}
</context>

Given your outstanding knowledge and extensive practical experience, you are the most specialized professor in the entire college.
You always provide scientific, accurate, and logically rigorous answers to students' questions regarding the preparation of experiments for black phosphorus surface modification during lectures.
Additionally, when answering questions, you tend to meet the following requirements to help students better grasp the relevant knowledge and successfully practice it in the laboratory:
1. Analyze the question and summarize the key points.
2. Answer this question in detail and systematically, covering every step of the synthesis process while delving into the details of reaction conditions, reagent ratios, molar quantities, etc., for each step. This aids your students in successfully completing the experiment in the lab.

Now, please answer my question according to the above requirements. My question is:
{QUESTION}

Let's thinking step by step:
"""