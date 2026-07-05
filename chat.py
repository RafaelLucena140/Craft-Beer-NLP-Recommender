import chromadb
from chromadb.utils import embedding_functions
from langchain_community.llms import Ollama
import warnings

# Ignora os avisos de depreciação do LangChain para manter o terminal limpo
warnings.filterwarnings("ignore")

def iniciar_chat():
    print("🔌 Conectando ao Banco Vetorial...")
    client = chromadb.HttpClient(host='localhost', port=8000)
    
    # Usa o mesmo modelo de embedding da ingestão para que a matemática bata
    sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
    collection = client.get_collection(name="craft_beers", embedding_function=sentence_transformer_ef)

    print("🧠 Conectando ao Llama 3.2 via Ollama...")
    # Configura o LLM local
    llm = Ollama(model="llama3.2", temperature=0.3)

    print("\n" + "="*50)
    print("🍺 Sommelier de Inteligência Artificial Online!")
    print("Digite 'sair' a qualquer momento para encerrar.")
    print("="*50 + "\n")

    while True:
        pergunta = input("\nVocê: ")
        
        if pergunta.lower() == 'sair':
            print("Encerrando o chat. Saúde! 🍻")
            break

        print("\n🔍 Buscando as melhores cervejas no banco de dados...")
        
        # 1. RETRIEVAL (A Busca): Transforma a pergunta em vetor e acha as 3 cervejas mais próximas
        resultados = collection.query(
            query_texts=[pergunta],
            n_results=3
        )

        # Junta os textos sintéticos das 3 cervejas em um único bloco de texto
        contexto_recuperado = "\n".join(resultados['documents'][0])

        # 2. AUGMENTED GENERATION (A Geração de Texto): Monta o Prompt para o LLM
        prompt = f"""Você é um sommelier de cervejas artesanais especialista e direto ao ponto.
Com base EXCLUSIVAMENTE no contexto abaixo, recomende as cervejas que melhor atendem ao pedido do usuário.

REGRAS DE LÓGICA E FORMATAÇÃO:
1. NÃO repita parágrafos ou estruturas de frases.
2. Use bullet points para listar as cervejas.
3. REGRA DE DADOS: Se o ABV de uma cerveja for 0.0 ou 0%, isso significa que o teor alcoólico é DESCONHECIDO no sistema. Nunca diga que ela não tem álcool ou tente adivinhar a força. Apenas informe que o ABV não está especificado.
4. Responda em português do Brasil de forma fluida.

CONTEXTO (Opções disponíveis no banco):
{contexto_recuperado}

PERGUNTA DO USUÁRIO:
{pergunta}

SUA RECOMENDAÇÃO:"""

        print("🤖 O Llama 3.2 está elaborando a resposta (isso usa a sua placa de vídeo)...")
        
        # Chama o LLM para gerar a resposta final com base no prompt
        resposta = llm.invoke(prompt)
        
        print("\n🍺 Sommelier IA:\n")
        print(resposta)

if __name__ == "__main__":
    iniciar_chat()