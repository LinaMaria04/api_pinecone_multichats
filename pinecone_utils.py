# C:\Proyectos\api_python\pinecone_utils.py

import os
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
#from langchain_openai import OpenAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from typing import List, Optional, Union
import traceback # Para print_exc()

load_dotenv()

# Variables globales para cachear las instancias 
_pinecone_client_instance = None
_embedding_model_instance = None 
_chat_model_instance = None 

# --- Funciones para obtener las instancias ---

# Inicializa y devulve el cliente de Pinecone si no ha sido creado aún.
def get_pinecone_client():
    global _pinecone_client_instance
    if _pinecone_client_instance is None:
        try:
            api_key = os.getenv("PINECONE_API_KEY")
            if not api_key:
                raise ValueError("PINECONE_API_KEY debe estar configurada.")
            _pinecone_client_instance = Pinecone(api_key=api_key) 
            print("DEBUG: Cliente de Pinecone (principal) inicializado correctamente.")
        except Exception as e:
            print(f"ERROR: No se pudo inicializar el cliente de Pinecone (principal): {e}")
            raise
    return _pinecone_client_instance

# Devuelve una especificación de Pinecone Serverless basada en las variables de entorno. 
def get_pinecone_spec():
    cloud = os.getenv("PINECONE_CLOUD", "aws")
    region = os.getenv("PINECONE_REGION", "us-east-1")
    print(f"DEBUG: Usando ServerlessSpec con cloud='{cloud}' y region='{region}'")
    return ServerlessSpec(cloud=cloud, region=region)

# Genera los embedding con HUGINGFACE EMBEDDING
def get_embedding_model():
    global _embedding_model_instance
    if _embedding_model_instance is None: # Solo inicializa si aún no existe
        HUGGINGFACE_EMBEDDING_MODEL_NAME = os.getenv("HUGGINGFACE_EMBEDDING_MODEL_NAME", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

        print(f"DEBUG: Intentando inicializar el modelo de embeddings de Hugging Face: {HUGGINGFACE_EMBEDDING_MODEL_NAME}")

        try:
            _embedding_model_instance = HuggingFaceEmbeddings(
                model_name=HUGGINGFACE_EMBEDDING_MODEL_NAME,
                encode_kwargs={'normalize_embeddings': True}
            )
            print(f"DEBUG: Modelo de Embeddings de Hugging Face '{HUGGINGFACE_EMBEDDING_MODEL_NAME}' inicializado correctamente.")
        except Exception as e:
            traceback.print_exc() 
            raise RuntimeError(f"Falló la inicialización del modelo de embeddings de Hugging Face: {e}. "
                               "Asegúrate de que el modelo exista, que `sentence-transformers` esté instalado y que haya suficiente RAM/CPU.")
    return _embedding_model_instance

# Crea una instancia del modelo de chat GPT (ChatOpenAI)
def get_openai_chat_model():
    global _chat_model_instance # Declara que vamos a usar la variable global
    if _chat_model_instance is None:
        openai_api_key = os.getenv("OPENAI_API_KEY")
        print(f"DEBUG: Intentando obtener OPENAI_API_KEY para ChatModel. Clave {'encontrada' if openai_api_key else 'NO encontrada'}")

        if not openai_api_key:
            raise ValueError("OPENAI_API_KEY no está configurada en las variables de entorno. Por favor, configúrala.")
        
        try:
            _chat_model_instance = ChatOpenAI( # Asigna a la variable global
                openai_api_key=openai_api_key, 
                model="gpt-3.5-turbo",
                temperature=0.7
            )
            print(f"DEBUG: Modelo de Chat OpenAI ('{_chat_model_instance.model_name}') inicializado correctamente.")
        except Exception as e:
            traceback.print_exc()
            raise RuntimeError(f"Falló la inicialización del modelo de chat de OpenAI: {e}. "
                               "Verifica tu clave de API y la conexión a OpenAI.")
    return _chat_model_instance # Devuelve la instancia cacheada


# Obtiene un vectorstore basado en el nombre del índice en Pinecone
def get_pinecone_vectorstore(index_name: str, namespace: Optional[Union[str, int]] = None):
    if not index_name:
        raise ValueError("Se requiere un nombre de indice para obtener el vectorstore en Pinecone")

    if index_name.startswith('vs_'):
        pinecone_friendly_index_name = index_name.replace('_', '-')
        print(f"DEBUG: Nombre de índice original de OpenAI: '{index_name}', convertido a: '{pinecone_friendly_index_name}' para LangChain/Pinecone.")
    else:
        pinecone_friendly_index_name = index_name 

    embeddings_model = get_embedding_model() 

    pinecone_namespace = str(namespace) if namespace is not None else None

    print(f"DEBUG: [pinecone_utils] Inicializando PineconeVectorStore para '{index_name}' con namespace '{pinecone_namespace}'.")
    
    # Usar el nombre convertido para interactuar con PineconeVectorStore
    vectorstore = PineconeVectorStore(
        index_name=pinecone_friendly_index_name, 
        embedding=embeddings_model,
        namespace=pinecone_namespace 
    )
    print(f"DEBUG: PineconeVectorStore para '{pinecone_friendly_index_name}' inicializado.")
    return vectorstore

# Inserta documentos (chunks) en un índice Pinecone
def upsert_documents_to_pinecone(documents: List[Document], index_name: str, name_space:Optional[Union[str, int]] = None):
    embeddings = get_embedding_model()

    target_namespace = None
    if name_space is not None:
        target_namespace = str(name_space)

    print(f"DEBUG: VectorStore existente para '{index_name}' inicializado.")
    # Carga la instancia del vectorstore existente usando el nombre del índice y el modelo de embeddings
    vectorstore = PineconeVectorStore.from_existing_index(index_name=index_name, embedding=embeddings)

    # Usa add_documents para añadir los documentos al namespace específico
    vectorstore.add_documents(
        documents=documents,
        namespace=target_namespace
    )
    print(f"DEBUG: Documentos (chunks) insertados con exito en el indice '{index_name}' en el namespace '{target_namespace}'.")