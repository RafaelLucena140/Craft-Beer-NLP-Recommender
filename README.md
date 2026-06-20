## 🏗️ Arquitetura (RAG Local)

O projeto está estruturado no modelo de Retrieval-Augmented Generation (RAG), garantindo privacidade de dados e zero custo com APIs externas:

1. **Engenharia de Dados (Scraping e ETL)**
   - Extração de catálogos e reviews de cervejas com `BeautifulSoup`/`Scrapy`.
   - Limpeza e chunking de texto.

2. **Base de Conhecimento Vetorial**
   - Geração de embeddings com modelos locais da Hugging Face.
   - Armazenamento em um Vector Database local (**ChromaDB** ou **Qdrant**).

3. **Geração e Orquestração (Local LLM)**
   - Orquestração do pipeline RAG utilizando **LangChain**.
   - Inferência rodando 100% localmente utilizando **Ollama** com modelos quantizados (ex: Llama 3 8B ou Mistral 7B).

4. **API e Interface**
   - Criação de uma API com **FastAPI** para servir as respostas do LLM.
   - Interface de Chat interativa desenvolvida em **Streamlit**.

## 🛠️ Stack Tecnológico
* **Linguagem:** Python
* **LLM Engine:** Ollama (Llama 3 / Mistral)
* **Vector DB e Orquestração:** ChromaDB, LangChain, HuggingFace Embeddings
* **Coleta de Dados:** `requests`, `BeautifulSoup`
* **Deploy/Frontend:** FastAPI, Streamlit, Docker