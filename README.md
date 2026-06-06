# 🍺 Craft Beer NLP Recommender

## 📌 Sobre o Projeto
Este é um projeto de Machine Learning de ponta a ponta focado em Processamento de Linguagem Natural (NLP) e Sistemas de Recomendação. O objetivo é ajudar consumidores a descobrirem novos rótulos de cervejas artesanais com base na similaridade semântica de perfis de sabor e notas de degustação.

O pipeline engloba todo o ciclo de vida do dado: desde a extração de dados brutos da web, passando pelo processamento matemático de texto, até o deploy do modelo em nuvem e a visualização executiva dos resultados.

## 🏗️ Arquitetura e Fases do Projeto

O projeto está estruturado em 5 fases principais:

1. **Aquisição de Dados e Engenharia de Dados**
   - Construção de rotinas de Web Scraping para extração de catálogo e reviews de usuários.
   - Ingestão e armazenamento de dados brutos em ambiente de nuvem (AWS S3).

2. **Pipeline de NLP (Pré-processamento)**
   - Limpeza e padronização de dados não estruturados (remoção de pontuação, *stopwords*, lematização).
   - Conversão do texto para formato vetorial utilizando técnicas de `TF-IDF` e Embeddings.

3. **Motor de Recomendação (Machine Learning)**
   - Implementação de um algoritmo de Filtragem Baseada em Conteúdo (*Content-Based Filtering*).
   - Utilização de Similaridade de Cosseno (*Cosine Similarity*) para identificar e ranquear as cervejas mais próximas no espaço vetorial.

4. **Engenharia de ML e Deploy (MLOps)**
   - Desenvolvimento de uma API RESTful de alta performance para servir o modelo.
   - Conteinerização de todo o ambiente da aplicação.
   - Deploy escalável na AWS.

5. **Consumo, Visualização e Analytics**
   - Criação de uma interface web interativa para testes do usuário final.
   - Integração com dashboards interativos no **Power BI** para análise de dados do catálogo (distribuição de estilos, análise de ABV/IBU, nuvem de palavras das avaliações).

## 🛠️ Stack Tecnológico
* **Linguagem Base:** Python e SQL
* **Coleta e Ingestão:** `requests`, `BeautifulSoup`
* **Manipulação e ML:** `pandas`, `scikit-learn`, `NLTK` / `spaCy`
* **Backend e MLOps:** `FastAPI`, Docker, AWS 
* **Front-end e BI:** `Streamlit`, Power BI

## 🚀 Como Executar o Projeto
*(Instruções de clonagem, criação de ambiente virtual e execução dos scripts serão adicionadas conforme o avanço das etapas de desenvolvimento).*

## 👨‍💻 Autor
Desenvolvido por **Rafael Gomes Lucena**.
