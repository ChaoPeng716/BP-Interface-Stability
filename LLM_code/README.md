# Black Phosphorus Surface Modification Information Extraction and QA Generation

## 🌟 Project Overview

This project is an automated information extraction and question-answering generation system specifically designed for processing academic literature related to Black Phosphorus surface modification. By combining large language models with text processing techniques, it automatically extracts key information from academic papers and generates high-quality QA datasets, providing intelligent knowledge management tools for black phosphorus materials research.

### Key Features
- 📄 **Literature Parsing**: Automatically parse academic papers in Markdown format
- 🔍 **Information Extraction**: Intelligently extract functional group information and experimental protocols
- 💬 **QA Generation**: Generate professional question-answer pairs based on extracted information
- 📊 **Dataset Construction**: Generate standardized training data

## 🚀 Quick Start

### Requirements
```bash
Python 3.10+
OpenAI API access
```

### Install Dependencies
```bash
pip install openai tqdm json glob multiprocessing
```

## 📁 Project Structure

```
├── task1.py                    # Functional group information extraction
├── task2.py                    # Experimental protocol extraction  
├── make_qa_bp.py              # Q&A pair generation
├── make_qa_prompts_bp.py      # Prompt templates
├── merge_qa2jsonl.py          # Dataset merging
└── README.md                  # Project documentation
```

## 🔧 Core Module Details

### [`task1.py`](task1.py) - Functional Group Information Extraction
Specifically designed for extracting functional group information related to black phosphorus surface modification from academic papers:

**Main Functions:**
- Text segmentation and classification
- Intelligent functional group information extraction
- JSON format output

**Core Functions:**
- `extract_info()`: Main extraction function
- `get_function_groups()`: Functional group identification
- `comfirm_json_string()`: JSON format repair

### [`task2.py`](task2.py) - Experimental Protocol Extraction
Extract specific experimental steps for black phosphorus surface modification from papers:

**Main Functions:**
- Experimental protocol extraction
- Method step parsing
- Structured data output

### [`make_qa_bp.py`](make_qa_bp.py) - QA Generation Engine
Generate high-quality question-answer pairs based on extracted information:

**Core Workflow:**
1. Candidate question generation
2. Question quality scoring
3. Optimal question selection
4. Professional answer generation