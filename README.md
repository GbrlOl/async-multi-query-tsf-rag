# **AsyncMultiQuery TSF-RAG: A Retrieval System for the Evaluation of the Physical Stability of Tailings Storage Facilities in Chile**

This repository contains all the codes needed to replicate the article. We recommend following the instructions for installation, initial configuration, etc.

## **Installation**

### **1. Clone the repository:**
```bash
git clone https://github.com/GbrlOl/async-multi-query-tsf-rag
cd async-multi-query-tsf-rag
```

### **2. Create and activate a virtual environment (recommended):**
```bash
# Using conda
conda create -n tsf-rag python=3.10
conda activate tsf-rag
```

#### Or using venv

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### **3. Install dependencies:**

```bash
pip install -r requirements.txt
```

### **4. Install the package in editable mode:**

```bash
pip install -e . --no-deps
```

## **Initial Configuration**

> [!WARNING]
> You need to have an API Key from OpenAI (LLM) and Nomic (Embedding). Without these resources, the systems will not work.

Once you have the API keys, you must create a `.env` file outside the src directory and inside the file you must have the API's as follows:

```bash
OPENAI_API_KEY=sk-...
NOMIC_API_KEY=nk-...
```

## **How to replicate the paper?**

To do this in a simple and educational way, we will provide you with a notebook file called `tsf_rag.ipynb`. Simply follow the proposed flow. In this notebook, you will find:

* How to use RAG systems?
* How can the experiments be replicated?

