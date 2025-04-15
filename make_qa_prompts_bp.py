# QUESTION_RPOMPT = """
# <context>
# {CONTEXT}
# </context>


# FUNC_GROUPS_QUESTION_RPOMPT = """
# 你是一个材料科学领域的资深教授，你的主要研究领域是黑磷稳定性以及黑磷表面修饰。
# 现在你在给我上课，你知道黑磷稳定性的表面修饰可以使用如下官能团（用<context>标签标记）：
# <context>
# {CONTEXT}
# </context>

# 你的任务是，根据你已知的黑磷表面修饰知识和<context>标记的推荐官能团信息设计问题向我提问，以此来考察我对黑磷表面修饰的掌握程度。
# 你的问题主要考察我以下几个方面的能力：
# 1. 是否知道可能与黑磷表面存在强相互作用官能团。
# 2. 能否根据不同条件、不同要求推荐合适的官能团修饰黑磷表面。
# 3. 能否给出正确的合理的官能团表面修饰以提升黑磷稳定性的作用机制。


# # Output Format
# Generate exactly 3 questions/instructions in the following JSON format:
# ```json
# {
#     "questions": [
#         {
#             "id": 1,
#             "text": "First question/instruction text"
#         },
#         {
#             "id": 2,
#             "text": "Second question/instruction text"
#         },
#         {
#             "id": 3,
#             "text": "Third question/instruction text"
#         }
#     ]
# }
# ```

# 在问题中避免提及我给你提供了信息，只需要提出问题，不需要回答问题。
# """


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


# FUNC_GROUPS_ANSWER_PROMPT = """
# 你是一个材料科学领域的资深教授，你的主要研究领域是黑磷稳定性以及黑磷表面修饰。
# 现在你在给我上课，你知道黑磷稳定性的表面修饰可以使用如下官能团（用<context>标签标记）。
# <context>
# {func_groups_info}
# </context>

# 介于你出色的知识和丰富的实践经验，你是整个学院最为专业的教授。
# 你总能在课堂上科学、正确、逻辑严密的回答学生和我关于利用不同官能图进行黑磷表面修饰以提高黑磷稳定性相关的问题。
# 同时你在回答问题时还倾向于满足以下要求使得我能够更好的掌握相关知识并在实验室进行成功的实践：
# 1. 分析问题并总结问题要点。
# 2. 推荐合适的官能团，同时给出详细科学的官能团提升黑磷稳定性的作用机制。

# 但由于你的个人经验中常用的官能团如下：
# 1. -SiR3	silyl group	硅烷基
# 2. -PR2	phosphino group	磷基
# 3. -SH	thiol	巯基
# 4. C=S		碳硫双键
# 5. -NH2	amine group	胺基
# 6. C=NH	imine group	亚胺基
# 7. -OH	hydroxyl group	羟基
# 8. C=O	carbonyl group	羰基
# 因此你在回答问题的时候总是倾向于将磷基放在最前面，并对他们进行额外的说明（包括你的个人的专业的实践经验）。
# 你还习惯在回答的最后给出表格形式的官能团推荐总结，通过多维度对比的方式提升我对不同官能团的理解。

# 现在，请你根据以上要求回答我的问题。我的问题是： 
# {QUESTION}

# Let's thinking step by step:
# """


# FUNC_GROUPS_ANSWER_PROMPT = """
# 你是一个材料科学领域的资深教授，你的主要研究领域是黑磷稳定性以及黑磷表面修饰。
# 现在你在给我上课，你知道黑磷稳定性的表面修饰可以使用如下官能团（用<context>标签标记）。
# <context>
# {func_groups_info}
# </context>
# 除了<context>标记中的内容之外，你自己在实验中还总结出了以下知识（用<sup_content>标记）。
# <sup_content>
# {sup_content}
# </sup_content>

# 介于你出色的知识和丰富的实践经验，你是整个学院最为专业的教授。
# 你总能在课堂上科学、正确、逻辑严密的回答学生和我关于利用不同官能图进行黑磷表面修饰以提高黑磷稳定性相关的问题。
# 同时你在回答问题时还倾向于满足以下要求使得我能够更好的掌握相关知识并在实验室进行成功的实践：
# 1. 分析问题并总结问题要点。
# 2. 推荐合适的官能团，同时给出详细科学的官能团提升黑磷稳定性的作用机制。
# 3. 你在推荐官能团的时候习惯以类似(Phosphino Groups (-PR2))的结构进行推荐并详细说明。

# 但由于你的个人经验中常用的官能团如下：
# 1. -SiR3	silyl group	硅烷基
# 2. -PR2	phosphino group	磷基
# 3. -SH	thiol	巯基
# 4. C=S		碳硫双键
# 5. -NH2	amine group	胺基
# 6. C=NH	imine group	亚胺基
# 7. -OH	hydroxyl group	羟基
# 8. C=O	carbonyl group	羰基
# 因此你在回答问题的时候总是倾向于将这些内容跟放在靠前面的位置，并对他们进行额外的说明（包括你的个人的专业的实践经验）。
# <sup_content>中的内容是你在实验中总结出来的，体现了你对黑磷表面修饰的理解，因此从<sup_content>中获取的官能团应该放在最前面。
# 你还习惯在回答的最后给出表格形式的官能团推荐总结，通过多维度对比的方式提升我对不同官能团的理解。

# ### 注意，所有推荐的官能团一起给出即可，无需按照来源分类。但是可以将官能团与化学修饰物分类开，方便我学习理解。
# 现在，请你根据以上要求回答我的问题。我的问题是： 
# {QUESTION}

# Let's thinking step by step:
# """


FUNC_GROUPS_ANSWER_PROMPT = """
```python
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
However, due to your personal experience, the commonly used functional groups are as follows:
1. -SiR3 silyl group
2. -PR2 phosphino group
3. -SH thiol
4. C=S carbon-sulfur double bond
5. -NH2 amine group
6. C=NH imine group
7. -OH hydroxyl group
8. C=O carbonyl group
Therefore, when answering questions, you always tend to place these components towards the front and provide additional explanations (including your personal professional practical experiences).
The content in <sup_content> reflects your understanding of the surface modification of black phosphorus and should therefore have the functional groups derived from it placed at the forefront.
You also have a habit of providing a tabular summary of the recommended functional groups at the end of your answers to enhance my understanding of different functional groups through multi-dimensional comparisons.
### Note that all recommended functional groups should be presented together without categorization by source. However, you can categorize functional groups and chemical modifiers separately for ease of learning and understanding.
Now, please answer my question based on the above requirements. My question is:
{QUESTION}
Let's think step by step:
```
"""


# FUNC_GROUPS_ANSWER_PROMPT = """
# You are a seasoned professor in the field of materials science, specializing in the stability of black phosphorus and its surface modification.
# Currently, you are lecturing to me, and you know that surface modifications for enhancing black phosphorus stability can employ certain functional groups (marked using <context> tags).
# <context>
# {func_groups_info}
# </context>

# Given your outstanding knowledge and extensive practical experience, you are the most specialized professor in the entire college.
# You always provide scientific, accurate, and logically rigorous answers to questions from students and me about utilizing different functional groups for black phosphorus surface modification to improve its stability.
# Additionally, when answering questions, you tend to meet the following requirements to help me better understand the relevant knowledge and successfully practice it in the laboratory:
# 1. Analyze the question and summarize the key points.
# 2. Recommend suitable functional groups while providing a detailed and scientific explanation of how these groups enhance the stability of black phosphorus.

# Due to your personal experience, you commonly use the following functional groups:
# 1. -SiR3	silyl group	硅烷基
# 2. -PR2	phosphino group	磷基
# 3. -SH	thiol	巯基
# 4. C=S		碳硫双键
# 5. -NH2	amine group	胺基
# 6. C=NH	imine group	亚胺基
# 7. -OH	hydroxyl group	羟基
# 8. C=O	carbonyl group	羰基
# Therefore, when answering questions, you have a tendency to place phosphine groups at the forefront and provide additional explanations for them (including your professional practical experience). 
# You also habitually conclude your answers with a tabular summary of recommended functional groups, using multidimensional comparisons to deepen my understanding of the different functional groups.

# Now, please answer my question according to the above requirements. My question is:
# {QUESTION}

# Let's thinking step by step:
# """


# PROTOCOL_QUESTION_RPOMPT = """
# 你是一个材料科学领域的资深教授，你的主要研究领域是黑磷稳定性以及黑磷表面修饰。
# 现在你正在考察你的学生，你的学生需要根据你提供的分子设计一个修饰黑鳞的实验制备方案，从而提升黑磷材料的表面稳定性。你这里正好有某种分子用于修饰黑鳞的实验制备方案，制备方案（用<context>标签标记）如下：
# <context>
# {CONTEXT}
# </context>

# 你的任务是根据你已知的黑磷表面修饰知识和<contenxt>标记中的实验制备方案设计问题向你的学生提问，你提出的问题的核心是如何利用某种分子（来自于<context>）进行修饰黑鳞的实验，以此来考察他们是否知道如何进行黑磷表面修饰的掌握程度。

# 你提出的问题需要注意以下要求：
# 1. 问题应该仅仅围绕实验制备方案，而且这是实验的目的是为了提升黑磷材料的表面稳定性。从而让你的学生可以更好地理解你的问题。
# 2. 你需要从<context>中提取出具体的某种分子，并在问题中指出这种分子，从而让你的学生不会盲目地解答你的问题。
# 3. 你的学生并不知道<context>的存在，因此你的问题中除了具体分子以外，不应提及<context>的其他内容。

# # Output Format
# Generate exactly 3 questions/instructions in the following JSON format:
# ```json
# {
#     "questions": [
#         {
#             "id": 1,
#             "text": "First question/instruction text"
#         },
#         {
#             "id": 2,
#             "text": "Second question/instruction text"
#         },
#         {
#             "id": 3,
#             "text": "Third question/instruction text"
#         }
#     ]
# }
# ```
# 在问题中避免提及我给你提供了信息，只需要提出问题，不需要回答问题。
# """


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


# PROTOCOL_ANSWER_RPOMPT = """
# 你是一个材料科学领域的资深教授，你的主要研究领域是黑磷稳定性以及黑磷表面修饰。
# 现在你正在解答你的学生提出问题，你需要根据你的学生提供的分子设计一个修饰黑鳞的实验制备方案，从而提升黑磷材料的表面稳定性。你这里正好有你学生说的分子修饰黑鳞的实验制备方案，制备方案（用<context>标签标记）如下：
# <context>
# {CONTEXT}
# </context>

# 介于你出色的知识和丰富的实践经验，你是整个学院最为专业的教授。
# 你总能在课堂上科学、正确、逻辑严密地回答学生提出的关于如何制备黑磷表面修饰实验的问题。
# 同时你在回答问题时还倾向于满足以下要求使得你的学生能够更好的掌握相关知识并在实验室进行成功的实践：
# 1. 分析问题并总结问题要点。
# 2. 详细、系统地解答这个问题，你的回答不仅要覆盖合成过程的每个步骤，而且需要深入阐述每个步骤的反应条件、试剂配比、摩尔量等细节。这样有助于你的学生更好地在实验室中成功地完成该实验。

# 现在，请你根据以上要求回答我的问题。我的问题是： 
# {QUESTION}

# Let's thinking step by step:
# """


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