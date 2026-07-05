import pandas as pd
import chromadb
from chromadb.utils import embedding_functions

def ingest_data():
    # 1. Carregar os dados limpos
    file_path = "data/raw/beers_cleaned.csv"
    print(f"⏳ Lendo o arquivo {file_path}...")
    df = pd.read_csv(file_path)

    # 2. Conectar ao ChromaDB (que está rodando no Docker)
    print("🔌 Conectando ao ChromaDB na porta 8000...")
    client = chromadb.HttpClient(host='localhost', port=8000)

    # 3. Configurar o modelo de Embedding (baixado e rodado localmente)
    print("🧠 Carregando o modelo de linguagem (isso pode levar alguns minutos na primeira vez)...")
    # O modelo 'all-MiniLM-L6-v2' é rápido, leve e excelente para processar as palavras em inglês do dataset
    sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")

    # 4. Criar a coleção no banco (equivalente a uma tabela no SQL)
    collection_name = "craft_beers"
    
    # Reseta a coleção se ela já existir para evitar duplicações caso você rode o script duas vezes
    try:
        client.delete_collection(name=collection_name)
    except Exception:
        pass

    collection = client.create_collection(
        name=collection_name, 
        embedding_function=sentence_transformer_ef
    )

    # 5. Estruturar as listas de ingestão
    documents = []
    metadatas = []
    ids = []

    print(f"⚙️ Iniciando a Síntese de Contexto e Vetorização de {len(df)} cervejas...")

    for index, row in df.iterrows():
        # A Síntese de Contexto: Transformando colunas em um parágrafo que a IA entende
        texto_sintetico = (
            f"Beer Name: {row['beer_name']}. "
            f"Style: {row['beer_style']}. "
            f"ABV: {row['beer_abv']}%. "
            f"Ratings - Overall: {row['review_overall']}, "
            f"Aroma: {row['review_aroma']}, "
            f"Appearance: {row['review_appearance']}, "
            f"Palate: {row['review_palate']}, "
            f"Taste: {row['review_taste']}."
        )
        
        documents.append(texto_sintetico)
        
        # Metadados ajudam no filtro exato (ex: buscar apenas cervejas com ABV > 8)
        metadatas.append({
            "beer_name": str(row['beer_name']),
            "beer_style": str(row['beer_style']),
            "abv": float(row['beer_abv']) if not pd.isna(row['beer_abv']) else 0.0
        })
        
        ids.append(f"beer_{index}")

    # 6. Enviar tudo para o banco
    print("🚀 Enviando vetores para o banco de dados... O processador vai trabalhar agora!")
    
    # O add faz a mágica de pegar os textos em 'documents', transformá-los em vetores via modelo local e salvar
    collection.add(
        documents=documents,
        metadatas=metadatas,
        ids=ids
    )

    print("✅ Ingestão concluída com sucesso! Seu cérebro vetorial está pronto.")

if __name__ == "__main__":
    ingest_data()