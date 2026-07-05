# 🍺 Sommelier de Cervejas IA (Local RAG)

Este projeto é um sistema de recomendação inteligente construído com arquitetura **Retrieval-Augmented Generation (RAG)**. Ele atua como um sommelier virtual capaz de sugerir cervejas artesanais com base em características sensoriais (aroma, paladar, aparência e estilo), garantindo privacidade total de dados e zero custo com APIs externas por rodar de forma 100% local.

## 🏗️ Arquitetura do Sistema

O pipeline foi desenhado com foco em melhores práticas de Engenharia de Machine Learning, dividido nas seguintes camadas:

*   **1. Engenharia de Dados (ETL e Contexto Sintético):**
    *   Limpeza e processamento de dados estruturados (milhares de avaliações de cervejas) utilizando `Pandas`.
    *   Tratamento de valores ausentes (ex: ABV desconhecido) no back-end para evitar alucinações do modelo.
    *   Transformação de colunas tabulares em "Textos Sintéticos" para maximizar a captura semântica durante a vetorização.

*   **2. Base de Conhecimento Vetorial:**
    *   Servidor de banco de dados vetorial **ChromaDB** conteinerizado e orquestrado via **Docker** (`docker-compose`), garantindo persistência em disco.
    *   Geração de *embeddings* matemáticos com modelos leves da Hugging Face (`sentence-transformers/all-MiniLM-L6-v2`) processados diretamente na CPU local.

*   **3. Geração e Orquestração (Local LLM):**
    *   Busca semântica avançada para recuperar o contexto ideal.
    *   Inferência de texto gerada pelo **Llama 3.2** rodando através do motor **Ollama**.
    *   Integração construída com `LangChain`, aplicando Engenharia de Prompt rigorosa e *Guardrails* de formatação para evitar o "efeito papagaio" e mascarar dados faltantes antes de chegarem ao usuário.

*   **4. Interface de Usuário (Front-End):**
    *   Painel web interativo desenvolvido puramente em **Streamlit**.
    *   Controles de MLOps expostos para o usuário final, como o ajuste dinâmico da *Temperatura* do modelo.
    *   Auditoria de linhagem de dados embutida (visualização dos metadados brutos recuperados pelo ChromaDB).

## 🛠️ Stack Tecnológico

*   **Linguagem:** Python
*   **LLM Engine:** Ollama (Llama 3.2)
*   **Vector DB:** ChromaDB (via Docker)
*   **Embeddings & NLP:** LangChain, HuggingFace (`sentence-transformers`)
*   **Processamento de Dados:** Pandas
*   **Frontend web:** Streamlit

## 💡 Exemplos de Uso no Chat

Ao executar a aplicação, você pode experimentar buscas que correlacionam estilo, sabor e sensações, como:
> *"Estou procurando uma Double IPA bem avaliada, de preferência com um perfil parecido com lúpulo Citra."*
> *"Quero uma cerveja forte e rústica, na linha de uma Brett Strong Ale, com um ABV elevado."*
> *"Recomende uma cerveja leve e refrescante com alta nota de paladar, perfeita para o verão."*

## 🚀 Como Executar o Projeto

1.  Suba o servidor do banco de dados vetorial:
    ```bash
    docker-compose up -d
    ```
2.  Faça a ingestão e vetorização do catálogo de cervejas:
    ```bash
    python data_pipeline/ingest_vectors.py
    ```
3.  Inicie a interface web do Sommelier:
    ```bash
    streamlit run app.py
    ```