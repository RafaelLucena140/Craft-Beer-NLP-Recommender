import pandas as pd
import os

def prepare_data():
    input_path = "data/raw/beers_dataset.csv"
    output_path = "data/raw/beers_cleaned.csv"
    
    print("⏳ Carregando o dataset bruto. Isso pode levar alguns segundos...")
    
    try:
        df = pd.read_csv(input_path)
    except FileNotFoundError:
        print(f"❌ Erro: O arquivo não foi encontrado em {input_path}")
        return

    print(f"✅ Arquivo carregado! Total de linhas originais: {len(df)}")
    
    # 1. Padronizar o nome das colunas (ajuste de acordo com o CSV que você baixou)
    # Assumindo as colunas clássicas do Kaggle: beer_name, beer_style, beer_abv, review_text
    colunas_desejadas = ['brewery_name','beer_name', 'beer_style', 'beer_abv', 'review_overall',
                         'review_aroma','review_appearance','review_palate','review_taste']
    
    # Filtra apenas as colunas que importam para a IA, se elas existirem no CSV
    colunas_presentes = [col for col in colunas_desejadas if col in df.columns]
    df = df[colunas_presentes]
    
    # 2. Limpeza de Dados Vazios (Drop NA)
    # Remove qualquer linha que não tenha texto de avaliação, pois o NLP precisa de texto
    df = df.dropna(subset=['review_overall','review_aroma','review_appearance','review_palate',
                           'review_taste'])
    
    # 3. Remover Duplicatas
    # Mantém apenas a primeira avaliação de cada cerveja para termos variedade
    df = df.drop_duplicates(subset=['beer_name'])
    
    # 4. Amostragem (Crucial para rodar localmente sem travar)
    # Vamos separar apenas as 5.000 primeiras cervejas únicas para o nosso banco vetorial
    tamanho_amostra = 5000
    if len(df) > tamanho_amostra:
        df = df.head(tamanho_amostra)
        
    print(f"🧹 Dados limpos e filtrados! Total de linhas para a IA: {len(df)}")
    
    # 5. Salvar o novo CSV limpo
    df.to_csv(output_path, index=False, encoding='utf-8')
    print(f"📁 Arquivo salvo pronto para vetorização em: {output_path}")

if __name__ == "__main__":
    prepare_data()