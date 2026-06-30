<<<<<<< HEAD
# GENCRE-RE
=======
# **GENCRE-RE: A Generative Creativity Framework for Requirements Engineering**

## **Overview**

GENCRE-RE is an automated framework designed to support creative requirements generation in Requirements Engineering (RE) using Large Language Models (LLMs). The framework combines semantic clustering, combinatorial feature recombination, and LLM-based requirement generation to produce novel, contextually appropriate, and high-quality software requirements.

The framework leverages GPT and Gemini models to generate candidate requirements and applies a uniqueness–quality-based selection process to identify representative requirements for evaluation.

## **Project Structure**

Artifacts/  
│  
├── data/  
│   ├── Raw data.txt  
│  
├── Implemented code/  
│   ├── gpt.py  
│   ├── gemini.py  
│    
├── Result/  
│   └── 1504 GPT IEEE.csv  
│   ├── 2126 Gemini IEEE.csv  
│   ├──gemini\_best\_matched\_to\_gpt.csv    
│   └──selected\_requirements\_per\_cluster.csv  
│── Evaluation/  
│   ├──57outlier\_cumulitive.ipynb.py  
│   └── 57 without outlier.py  
│  
├── survey/  
│     
└── README.md

## **Prerequisites**

### **Software Requirements**

* Python 3.10 or later  
* pip  
* OpenAI API Key  
* Google Gemini API Key

### **Recommended Environment**

Ubuntu 22.04+

or

Windows 10/11

---

## **Installation**

### **Step 1: Download the Repository**

Download and unzip the file

### **Step 2: Create a Virtual Environment**

Linux/Mac:

python3 \-m venv venv

source venv/bin/activate

Windows:

python \-m venv venv

venv\\Scripts\\activate

### **Step 3: Install Dependencies**

pip install \-r requirements.txt

---

## **Required Python Libraries**

pip install pandas  
pip install numpy  
pip install scikit-learn  
pip install sentence-transformers  
pip install bertopic  
pip install umap-learn  
pip install hdbscan  
pip install openai  
pip install google-generativeai  
pip install nltk  
pip install tqdm

---

## **API Configuration**

Create a `.env` file in the project root directory.

OPENAI\_API\_KEY=your\_openai\_api\_key

GEMINI\_API\_KEY=your\_gemini\_api\_key

Example Python configuration:

import os

OPENAI\_API\_KEY \= os.getenv("OPENAI\_API\_KEY")  
GEMINI\_API\_KEY \= os.getenv("GEMINI\_API\_KEY")

---

## **Input Dataset Preparation**

Place the original requirements dataset inside:

data/Raw data.txt/

Accepted format:

requirement\_id,requirement  
1,The system shall allow users to create meetings.  
2,The system shall allow users to schedule webinars.  
...

## **Code Implementation**

Run the code files/Implemented code/[gpt.py](http://gpt.py) and /Implemented code/[gemini.py](http://gemini.py) and you will get all the resultant files.

>>>>>>> 7dab479 (First commit)
