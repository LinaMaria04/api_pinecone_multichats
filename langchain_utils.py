# python_api/langchain_utils.py
from langchain.chains import RetrievalQA
from langchain.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
import pinecone_utils
from typing import Optional 

def get_retrieval_qa_chain(index_name: str, system_behavior: Optional[str] = None): 
    
    if not index_name:
        raise ValueError("Se requiere un nombre de índice para obtener la cadena de RAG.")
    
    """
    Configura y devuelve una cadena de RAG (Retrieval Augmented Generation) de LangChain.
    Esta cadena se encarga de:
    1. Buscar información relevante en Pinecone (el "Retrieval").
    2. Usar esa información para generar una respuesta más precisa con un LLM (la "Generation").
    """
    llm = pinecone_utils.get_openai_chat_model()
    vectorstore = pinecone_utils.get_pinecone_vectorstore(index_name)

    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    #Definir el Prompt:
    base_template = """Eres un asistente amigable y útil. Responde a la pregunta basándote únicamente en el siguiente contexto:
    {context}

    Si la respuesta no está en el contexto, di que no tienes suficiente información para responder.
    Pregunta: {question}
    """
    
    # Si se proporciona un comportamiento de sistema, lo añadimos al inicio del template.
    if system_behavior:
        template = f"{system_behavior}\n\n{base_template}"
    else:
        template = base_template

    prompt = ChatPromptTemplate.from_template(template)

    # ... el resto de la función es igual ...
    def format_docs(docs: list[Document]) -> str:
        return "\n\n---\n\n".join(doc.page_content for doc in docs)

    rag_chain = (
        {"context": retriever | RunnableLambda(format_docs), "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    print(f"DEBUG: rag_chain antes de retornar en get_retrieval_qa_chain: {rag_chain}")
    return rag_chain