"""Streamlit user interface for the local beer recommender."""

import logging

import streamlit as st

from recommender import available_styles, connect_collection, generate_answer, retrieve_beers

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

st.set_page_config(page_title="Sommelier de Cervejas IA", page_icon="🍺", layout="centered")


@st.cache_resource
def load_recommender():
    collection = connect_collection()
    return collection, available_styles(collection)


try:
    collection, styles = load_recommender()
    connection_error = None
except Exception as exc:
    logger.exception("Não foi possível inicializar a busca vetorial")
    collection, styles = None, []
    connection_error = exc

st.title("🍺 Sommelier de Cervejas IA")
st.markdown("Peça recomendações por estilo, sabor, teor alcoólico ou notas da comunidade.")

st.sidebar.header("Configurações")
temperature = st.sidebar.slider("Criatividade da resposta", min_value=0.1, max_value=1.0, value=0.3, step=0.1)
st.sidebar.caption("A busca e os filtros são aplicados antes da geração da resposta.")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if question := st.chat_input("O que você gostaria de beber?"):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        if connection_error:
            st.error("Não consegui acessar o catálogo vetorial. Confira o container ChromaDB e tente novamente.")
        else:
            try:
                with st.spinner("Buscando e ordenando cervejas compatíveis..."):
                    result = retrieve_beers(collection, question, styles)
                if not result["items"]:
                    answer = "Não encontrei cervejas que atendam aos filtros desta consulta no catálogo atual."
                    st.info(answer)
                else:
                    with st.spinner("Preparando a recomendação..."):
                        answer = generate_answer(question, result["items"], temperature)
                    st.markdown(answer)
                with st.expander("Ver filtros e dados recuperados"):
                    st.write("Filtros aplicados:", result["filters"] or "Busca sem filtros estruturados")
                    for item in result["items"]:
                        metadata = item["metadata"]
                        st.markdown(
                            f"**{metadata.get('beer_name', 'Cerveja')}** — {metadata.get('beer_style', 'Estilo desconhecido')}"
                        )
                        st.caption(
                            f"ABV: {metadata['abv']:.1f}%" if metadata.get("abv_known") else "ABV não especificado"
                        )
                        st.caption(
                            f"Nota geral média: {metadata.get('review_overall', 0):.2f}/5 · "
                            f"Avaliações: {metadata.get('review_count', 'n/d')} · "
                            f"Dispersão da nota: {metadata.get('review_overall_std', 0):.2f}"
                        )
                        st.code(item["document"], language="text")
                st.session_state.messages.append({"role": "assistant", "content": answer})
            except Exception:
                logger.exception("Falha ao processar recomendação")
                st.error("Ocorreu um erro ao buscar ou gerar a recomendação. Confira os serviços locais e tente novamente.")
