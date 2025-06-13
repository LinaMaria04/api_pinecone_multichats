# python_api/langchain_utils.py
from langchain.chains import RetrievalQA
from langchain.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
import pinecone_utils
from typing import Optional, Union
from operator import itemgetter
from langchain.chains.combine_documents import create_stuff_documents_chain

def get_retrieval_qa_chain(
    index_name: str,
    system_behavior: Optional[str] = None,
    chat_id: Optional[Union[str, int]] = None, # <-- Mantén Union[str, int]
):
    if not index_name:
        raise ValueError("Se requiere un nombre de índice para obtener la cadena de RAG.")

    llm = pinecone_utils.get_openai_chat_model()

    # --- INICIO DEPURACIÓN DE NAMESPACE Y VECTORSTORE ---
    query_namespace = None
    if chat_id is not None and str(chat_id).lower() != "none":
        query_namespace = str(chat_id) # Convertir a string para usar como namespace
        print(f"DEBUG: [langchain_utils] Namespace FINAL usado para PineconeVectorStore: '{query_namespace}'")
    else:
        print("DEBUG: [langchain_utils] Namespace FINAL usado para PineconeVectorStore: POR DEFECTO (None)")

    # MODIFICACIÓN CLAVE: Llama a get_pinecone_vectorstore pasando el namespace
    try:
        vectorstore = pinecone_utils.get_pinecone_vectorstore(index_name=index_name, namespace=query_namespace)
        print(f"DEBUG: [langchain_utils] PineconeVectorStore para '{index_name}' inicializado con namespace '{query_namespace}'.")
    except Exception as e:
        print(f"ERROR: [langchain_utils] No se pudo inicializar VectorStore desde índice existente '{index_name}' con namespace '{query_namespace}': {e}")
        raise # Propagar el error


    # --- VERIFICACIÓN DE ESTADÍSTICAS DEL ÍNDICE EN CONSULTA ---
    try:
        pc_client = pinecone_utils.get_pinecone_client() # Asegúrate de que esta función exista y devuelva un cliente de Pinecone
        index = pc_client.Index(index_name)
        index_stats = index.describe_index_stats()
        print(f"DEBUG: [langchain_utils] Estadísticas del índice '{index_name}' ANTES DE LA BÚSQUEDA:")
        print(f"{json.dumps(index_stats.to_dict(), indent=2)}")

        # Verifica el namespace usado (si estás usando el enfoque de namespace directo)
        ns_to_check = query_namespace if query_namespace else '' # Si es None, verifica el default ('')
        if ns_to_check in index_stats.namespaces:
            count_in_namespace = index_stats.namespaces[ns_to_check].vector_count
            print(f"DEBUG: [langchain_utils] VERIFICACIÓN: El namespace '{ns_to_check}' EXISTE y tiene {count_in_namespace} vectores.")
        else:
            print(f"DEBUG: [langchain_utils] VERIFICACIÓN: El namespace '{ns_to_check}' NO EXISTE en las estadísticas del índice o está vacío.")
    except Exception as e:
        print(f"ERROR: [langchain_utils] Falló la verificación de estadísticas del índice en get_retrieval_qa_chain: {e}")
    # --- FIN VERIFICACIÓN DE ESTADÍSTICAS ---

    # ELIMINAR EL FILTRO DE METADATOS DEL RETRIEVER
    # Porque el vectorstore ya está filtrando por namespace
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5}) # Sin filtro aquí

    base_template = """Eres un asistente amigable y útil. Responde a la pregunta basándote únicamente en el siguiente contexto:
    {context}

    Si la respuesta no está en el contexto, di que no tienes suficiente información para responder.
    Pregunta: {question}
    """

    if system_behavior:
        template = f"{system_behavior}\n\n{base_template}"
    else:
        template = base_template

    qa_prompt = ChatPromptTemplate.from_template(template)

    document_chain = create_stuff_documents_chain(llm, qa_prompt)

    # --- AÑADIR UN PASO INTERMEDIO PARA VER LOS DOCUMENTOS RECUPERADOS ---
    def log_and_return_docs(docs: list[Document]):
        print("\n--- DOCUMENTOS RECUPERADOS POR EL RETRIEVER (LANGCHAIN_UTILS) ---")
        if not docs:
            print("¡El retriever NO recuperó documentos!")
        else:
            print(f"Se recuperaron {len(docs)} documentos.")
            for i, doc in enumerate(docs):
                print(f"--- Documento {i+1} ---")
                print(f"  Contenido: {doc.page_content[:500]}...") # Imprime los primeros 500 caracteres
                print(f"  Metadatos: {doc.metadata}") # MUY IMPORTANTE: Ver la metadata aquí
                print("---------------------")
        return docs

    rag_chain = (
        {"context": itemgetter("question") | retriever | RunnableLambda(log_and_return_docs), # REINCORPORADO
         "question": RunnablePassthrough()
        }
        | document_chain # Usa document_chain aquí, no qa_prompt | llm | StrOutputParser()
    )

    print(f"DEBUG: RAG chain (without history) initialized and ready.")

    return {"qa_chain": rag_chain, "retriever": retriever}
