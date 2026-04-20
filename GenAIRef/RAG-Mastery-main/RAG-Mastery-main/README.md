# 🧠 RAG-Mastery

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=for-the-badge&logo=jupyter&logoColor=white)
![LangChain](https://img.shields.io/badge/LangChain-1C3C3C?style=for-the-badge&logo=langchain&logoColor=white)
![LangGraph](https://img.shields.io/badge/LangGraph-FF4B4B?style=for-the-badge)
![RAG](https://img.shields.io/badge/RAG-Retrieval_Augmented_Generation-0A66C2?style=for-the-badge)

**A complete end-to-end learning repository** covering **Retrieval-Augmented Generation (RAG)** from fundamentals to advanced concepts.

This repository is built by following and extending the **"LangChain & RAG Mastery"** Udemy course by **Krish Naik**.

It includes all the concepts taught in the course along with my personal notes, improvements, and additional experiments.

---

## ✨ What You'll Learn

- Core RAG architecture and components
- Data ingestion, parsing, and advanced chunking strategies
- Vector embeddings, hybrid search, and query enhancement
- Multimodal RAG
- Building AI Agents and Agentic RAG
- LangGraph for complex workflows
- Multi-agent systems, Corrective RAG, Adaptive RAG
- Persistent memory, caching, and RAG evaluation
- Graph Databases with Cypher queries using LangChain

---

## 📁 Project Structure

```
RAG-Mastery/
├── 01. Introduction to RAG/
├── 02. Core Components in RAG/
├── 03. Data Ingestion and Data Parsing Techniques/
├── 04. Vector Embeddings and Vector Databases/
├── 05. Vector Stores and Vector Databases/
├── 06. Advance Chunking and Preprocessing Techniques/
├── 07. Hybrid Search Strategies/
├── 08. Query Enhancement/
├── 09. MultiModal Introduction and Multi-Model RAG/
├── 10. Introduction to AI Agents and Agentic AI/
├── 11. Updated Langchain Hands On with Version V1/
├── 12. LangGraph Basics/
├── 13. Agents Architecture/
├── 14. Agentic RAG/
├── 15. Autonomous RAG/
├── 16. Multi Agents RAGS/
├── 17. Corrective RAG/
├── 18. Adaptive RAG/
├── 19. RAG With Persistant Memory/
├── 20. Cache RAG With LangGraph/
├── 21. ChatBot and RAG Evaluation/
├── 22. Introduction to Graph Databases and Cypher Query Language with Langchain/
├── 23. Practical Implementation with GraphDB with Langchain/
├── .devcontainer/
├── requirements.txt
├── .gitignore
└── README.md
```

Each folder contains well-commented Jupyter notebooks (`.ipynb`) for hands-on learning.

---

## 🚀 Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/mani24singh/RAG-Mastery.git
cd RAG-Mastery
```

### 2. Create Virtual Environment (Recommended with uv)

```bash
# Install uv
pip install uv

# Create virtual environment
uv venv --python 3.11.13

# Activate the environment
# Windows:
.venv\Scripts\activate
# macOS / Linux:
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
uv pip install -r requirements.txt
```

### 4. Register Jupyter Kernel

```bash
uv pip install ipykernel

uv run python -m ipykernel install --user --name "RAG-Mastery" --display-name "RAG-Mastery (.venv)"
```

> **Tip**: Close and reopen VS Code after registering the kernel, then select **"RAG-Mastery (.venv)"** as the notebook kernel.

---

## 🔑 Environment Variables

Create a `.env` file in the root directory:

```env
# LLMs
OPENAI_API_KEY="your-openai-key"
GROQ_API_KEY="your-groq-key"
GOOGLE_API_KEY="your-google-key"

# Frameworks
LANGSMITH_API_KEY="your-langsmith-key"

# Vector Databases
PINECONE_API_KEY="your-pinecone-key"
ASTRA_DB_API_ENDPOINT="your-astra-endpoint"
ASTRA_DB_APPLICATION_TOKEN="your-astra-token"

# AI Agents
TAVILY_API_KEY="your-tavily-key"
```

> `.env` is already added to `.gitignore`.

---

## 🙏 Special Thanks

This repository is created while learning from the **"LangChain & RAG Mastery"** Udemy course by **Krish Naik**.  

Huge thanks to **Krish Naik** for creating such a comprehensive and practical course on RAG and LangChain.

---

## 🤝 Contributing

Contributions, improvements, and suggestions are welcome!  
Feel free to fork the repo and submit a pull request.

---

## 📜 License

This project is licensed under the **MIT License**.

---

## ⭐ Show Your Support

If this repository helped you in mastering RAG concepts, please give it a ⭐ on GitHub!  

Your support encourages me to keep improving and adding more content.

---

**Happy Learning & Building Powerful RAG Systems!** 🚀

Made with ❤️ by [Mani](https://github.com/mani24singh)

