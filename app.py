import streamlit as st
import chromadb
from chromadb.utils import embedding_functions
from langchain_community.llms import Ollama
import warnings

warnings.filterwarnings("ignore")

# Configuração da página do Streamlit
st.set_page_config(
    page_title="Sommelier de Cervejas IA",
    page_icon="🍺",
    layout="centered"
)

# Inicialização das conexões (usando cache para não reconectar a cada clique)
@st.cache_resource
def iniciar_conexoes():
    # Conecta ao ChromaDB no Docker
    client = chromadb.HttpClient(host='localhost', port=8000)
    sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
    collection = client.get_collection(name="craft_beers", embedding_function=sentence_transformer_ef)
    return collection

try:
    collection = iniciar_conexoes()
    conexao_ok = True
except Exception as e:
    conexao_ok = False

# Interface Visual
st.title("🍺 Sommelier de Cervejas IA")
st.markdown("Bem-vindo ao seu especialista em cervejas artesanais. Pergunte sobre estilos, aromas ou peça uma recomendação sob medida!")

# Barra Lateral de Configurações
st.sidebar.header("⚙️ Configurações da IA")
temperatura = st.sidebar.slider("Temperatura do Modelo", min_value=0.1, max_value=1.0, value=0.3, step=0.1)
st.sidebar.markdown("""
*💡 **Dica de MLOps:** Temperaturas baixas (0.1 - 0.3) deixam o sommelier focado e preciso. Temperaturas altas liberam a criatividade do modelo.*
""")

# Inicializa o histórico de chat na sessão do Streamlit
if "messages" not in st.session_state:
    st.session_state.messages = []

# Exibe as mensagens anteriores do histórico
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Campo de entrada do Chat
if pergunta := st.chat_input("O que você está a fim de beber hoje?"):
    
    # Exibe a pergunta do usuário
    with st.chat_message("user"):
        st.markdown(pergunta)
    st.session_state.messages.append({"role": "user", "content": pergunta})

    if not conexao_ok:
        with st.chat_message("assistant"):
            st.error("❌ Erro de conexão: Garanta que o contêiner do ChromaDB no Docker está rodando.")
    else:
        # Fluxo RAG
        with st.spinner("🔍 Consultando o cérebro vetorial..."):
            resultados = collection.query(query_texts=[pergunta], n_results=3)
            contexto_recuperado = "\n".join(resultados['documents'][0])

            # TÉCNICA DE MLOPS: Sanitização forçada do texto antes de ir para o LLM
            contexto_recuperado = contexto_recuperado.replace("nan%", "Desconhecido")
            
        with st.spinner("🧠 Elaborando recomendação com Llama 3.2..."):
            # Configura o LLM com a temperatura escolhida no slider
            llm = Ollama(model="llama3.2", base_url="http://127.0.0.1:11434", temperature=temperatura)
            
            prompt = f"""Você é um sommelier de cervejas artesanais especialista e direto ao ponto.
Com base EXCLUSIVAMENTE no contexto abaixo, recomende as cervejas que melhor atendem ao pedido do usuário.

REGRAS DE LÓGICA E FORMATAÇÃO:
1. NÃO repita parágrafos ou estruturas de frases.
2. Use bullet points para listar as cervejas.
3. REGRA DE DADOS: Se o ABV de uma cerveja constar como 0.0, 0% ou nan%, isso significa que o teor alcoólico é DESCONHECIDO. NUNCA exiba a palavra 'nan'. Apenas informe textualmente que o ABV não está especificado.
4. Responda em português do Brasil de forma fluida.

CONTEXTO (Opções disponíveis no banco):
{contexto_recuperado}

PERGUNTA DO USUÁRIO:
{pergunta}

SUA RECOMENDAÇÃO:"""

            resposta = llm.invoke(prompt)

        # Exibe a resposta do Sommelier
        with st.chat_message("assistant"):
            st.markdown(resposta)
            
            # Expander opcional para auditoria de dados (estilo linhagem de dados)
            with st.expander("🛠️ Ver metadados recuperados (RAG Context)"):
                st.code(contexto_recuperado, language="text")
                
        st.session_state.messages.append({"role": "assistant", "content": resposta})