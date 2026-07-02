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

> [!IMPORTANT]
> Using API LLM (OpenAI) as an evaluator can generate variability in the response, even if we have temperature=0. Keep that in mind!

# **Citation** 

```bibtex
@article{OLMOS2026133465,
title = {AsyncMultiQuery TSF-RAG: Improving Retrieval Robustness in Spanish Technical Reports for Tailings Storage Facility Physical Stability Evaluation},
journal = {Expert Systems with Applications},
pages = {133465},
year = {2026},
issn = {0957-4174},
doi = {https://doi.org/10.1016/j.eswa.2026.133465},
url = {https://www.sciencedirect.com/science/article/pii/S0957417426023742},
author = {Gabriel Olmos and Gabriel Villavicencio and Gabriel Hermosilla and Giovanni Cocca-Guardia and Manuel Silva and Vinicius Minatogawa and Pierre Breul},
keywords = {Retrieval-augmented generation, Embeddings, Large language model, Tailings storage facilities, Physical stability},
abstract = {Physical stability assessment of tailings storage facilities (TSFs) requires locating critical parameters in long and heterogeneous technical reports, a task currently performed manually in Chilean regulatory inspection workflows. Although Retrieval-Augmented Generation (RAG) systems can support technical document search, their performance is strongly affected by the retrieval stage, especially in specialized Spanish documentation. This study proposes AsyncMultiQuery TSF-RAG, a RAG system that generates multiple query reformulations and processes them asynchronously using different embedding models to improve retrieval robustness. The system was evaluated using 29 real Chilean mining reports covering five TSF typologies and 1,949 pages, with expert-validated ground truths for embedding retrieval and RAG response evaluation. Results show that standalone embedding models achieved low retrieval performance, with a maximum F1 Score of 0.2035. In the expert-validated evaluation, AsyncMultiQuery TSF-RAG achieved success rates between 66.67% and 75%, reducing complete parameter search time from 420–600 minutes to 5.62–11.35 minutes. These findings position AsyncMultiQuery TSF-RAG as a viable support tool for expert systems in regulatory auditing workflows.}
}
```
