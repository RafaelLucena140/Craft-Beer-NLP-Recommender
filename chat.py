"""Command-line chat using the same retrieval path as the Streamlit app."""

import logging

from recommender import available_styles, connect_collection, generate_answer, retrieve_beers

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def iniciar_chat():
    try:
        collection = connect_collection()
        styles = available_styles(collection)
    except Exception:
        logger.exception("Não foi possível conectar ao ChromaDB")
        print("Não foi possível conectar ao catálogo. Confira o ChromaDB e as configurações.")
        return

    print("Sommelier local pronto. Digite 'sair' para encerrar.")
    while True:
        question = input("\nVocê: ").strip()
        if question.casefold() == "sair":
            print("Até a próxima. Saúde!")
            break
        if not question:
            continue
        try:
            result = retrieve_beers(collection, question, styles)
            if not result["items"]:
                print("Não encontrei cervejas que atendam aos filtros desta consulta no catálogo atual.")
                continue
            print("Filtros: " + (", ".join(result["filters"]) if result["filters"] else "nenhum filtro estruturado"))
            answer = generate_answer(question, result["items"])
            print("\nSommelier IA:\n" + answer)
        except Exception:
            logger.exception("Falha ao processar consulta")
            print("Ocorreu um erro ao buscar ou gerar a recomendação. Confira os serviços locais e tente novamente.")


if __name__ == "__main__":
    iniciar_chat()
