# python_api/main.py
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from pydantic import BaseModel
from langchain_core.documents import Document
from typing import Optional, Dict, Any, List
import pinecone_utils
import langchain_utils
import config
import os
import io
import json # Necesario para json.loads()
import pypdf # Necesario para extraer texto de PDFs
import traceback 
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter 
from langchain_openai import OpenAIEmbeddings 
from langchain_pinecone import PineconeVectorStore
load_dotenv()


# Inicializa FastAPI
app = FastAPI(
    title="Chatbot Vector DB API con LangChain",
    description="API para interactuar con Pinecone, OpenAI y LangChain para tu chatbot PHP."
)

# Modelos Pydantic para la validación de datos de entrada
class DataToUpsert(BaseModel):
    id: str
    text_content: str
    metadata: dict[str, Any] #Metadatos adicionales para el documento
    vectorstore_id: str #ID del vectorstore (Nombre del índice en Pinecone)

class ChatRequest(BaseModel):
    question: str
    vectorstore_id: str # ID del vectorstore (Nombre del índice en Pinecone)
    comportamiento_chat: str = "" 
    chat_history: list = [] # Campo opcional si decides enviar historial de chat

# Evento de inicio de la API
@app.on_event("startup")
async def startup_event():
    try:
        pinecone_utils.get_pinecone_client()
        print("API de Pinecone y LangChain inicializadas correctamente.")
    except Exception as e:
        print(f"Error al iniciar la API: {e}")
        
# --- Rutas de la API ---
@app.get("/")
async def root():
    return {"message": "¡API de Chatbot con Pinecone, OpenAI y LangChain activa!"}

#Enviar texto plano para indedxarlo en Pinecone
@app.post("/ingest_data")
async def ingest_data(data: DataToUpsert):
    try:
        document = Document(
            page_content=data.text_content,
            metadata={**data.metadata, "id": data.id}
        )
        print(f"DEBUG: vectorstore_id que se usará para upsert: {data.vectorstore_id}") 
        pinecone_utils.upsert_documents_to_pinecone([document], data.vectorstore_id)
        
        return {"status": "success", "message": f"Documento con ID '{data.id}' ingestdo en Pinecone en el índice '{data.vectorstore_id}'."}
    except Exception as e:
        import traceback
        traceback.print_exc()  # Esto te ayudará a ver cualquier futuro error inesperado
        raise HTTPException(status_code=500, detail=f"Error al ingesttar datos: {e}")

#Sube archivos (PDF, TXT, JSON), los transforma en texto, los divide en chunks y los procesa para Pinecone
@app.post("/ingest_file")
async def ingest_file(
    file: UploadFile = File(...),
    document_id: str = Form(...),
    metadata_json: str = Form(...),
    vectorstore_id: str = Form(...),
):
    try:
        metadata = json.loads(metadata_json)
        file_content = await file.read()
        text_content = ""

        if file.content_type == "application/pdf":
            try:
                reader = pypdf.PdfReader(io.BytesIO(file_content))
                for page_num, page in enumerate(reader.pages):
                    text_content += page.extract_text() or ""
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"Error al extraer texto del PDF: {e}")
        elif file.content_type == "text/plain":
            text_content = file_content.decode('utf-8')
        elif file.content_type == "application/json":
            try:
                json_data = json.loads(file_content.decode('utf-8'))
                text_content = json.dumps(json_data, ensure_ascii=False, indent=2)
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"Error al procesar JSON: {e}")
        else:
            raise HTTPException(status_code=400, detail="Tipo de archivo no soportado para ingesta directa.")

        if not text_content.strip():
            raise HTTPException(status_code=400, detail="El archivo no contiene texto extraíble o está vacío después de la extracción.")

        # LÓGICA DE CHUNKING AQUÍ!
        #Crear un Documento base con todo el texto
        base_document = Document(page_content=text_content, metadata={
            "original_file_name": file.filename, # Nombre original del archivo
            "document_id": document_id, # ID único que envías desde Laravel
            **metadata # fusionar metadatos enviados desde Laravel
        })

        #Inicializar el Text Splitter
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,  # Tamaño recomendado de chunks
            chunk_overlap=200, # Solapamiento entre chunks
            length_function=len,
            add_start_index=True, # Añade el índice inicial del chunk en el documento original
        )

        #Dividir el documento grande en chunks
        chunks = text_splitter.split_documents([base_document])
        print(f"DEBUG: El archivo original se dividió en {len(chunks)} chunks.")

        #Limpiar o añadir metadatos específicos para cada chunk
        processed_chunks = []
        for i, chunk in enumerate(chunks):
            # Copiar metadatos existentes y añadir o sobrescribir si es necesario
            new_metadata = chunk.metadata.copy()
            new_metadata["chunk_index"] = i # Índice del chunk
                       
            processed_chunks.append(Document(page_content=chunk.page_content, metadata=new_metadata))

        #Llamar a la función de ingesta de Pinecone con los CHUNKS
        print(f"DEBUG: Ingestando {len(processed_chunks)} chunks en el vectorstore '{vectorstore_id}'...")
        pinecone_utils.upsert_documents_to_pinecone(processed_chunks, vectorstore_id)
        print(f"DEBUG: Documento con ID '{document_id}' ingestado exitosamente (chunks).")

        return {"message": "Archivo y datos ingestados exitosamente", "document_id": document_id, "chunks_ingested": len(processed_chunks)}

    except HTTPException as e:
        raise e
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error interno del servidor al procesar el archivo: {e}")


#Recibe y responde preguntas al ususario
@app.post("/ask_chatbot")
async def ask_chatbot(request: ChatRequest):
    try:
        chain_components = langchain_utils.get_retrieval_qa_chain(
            index_name=request.vectorstore_id, 
            system_behavior=request.comportamiento_chat
        )
        qa_chain = chain_components["qa_chain"]
        retriever = chain_components["retriever"] 
        
        docs = retriever.invoke(request.question) 
        
        print("\n--- DOCUMENTOS RECUPERADOS PARA LA PREGUNTA ---")
        print(f"Pregunta del usuario: {request.question}") 
        for i, doc in enumerate(docs):
            print(f"Documento {i+1}:")
            print(f"  Contenido: {doc.page_content[:500]}...")
            if doc.metadata:
                print(f"  Metadatos: {doc.metadata}")
            print("-" * 20)
        print("------------------------------------------\n")
        
        result = qa_chain.invoke({"question": request.question}) 
        response_content = result 
        
        return {"response": response_content}

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error al procesar la pregunta del chatbot: {e}")
 

#Realizar busquedad semnatica en Pinecone usando Simiularity Search
@app.get("/query_pinecone_index")   
async def query_pinecone_index(index_name: str, query_text: str = "documento", top_k: int = 5):
    #Consulta el índice de Pinecone con un texto de consulta y devuelve los documentos relevantes con sus metadatos.
    try:
        vectorstore = pinecone_utils.get_pinecone_vectorstore(index_name)
        
        print(f"DEBUG: Realizando búsqueda de similitud para '{query_text}' en índice '{index_name}' con k={top_k}")
        
        relevant_docs: List[Document] = vectorstore.similarity_search(query_text, k=top_k)
        
        if not relevant_docs:
            return {"message": "No se encontraron documentos relevantes para la consulta.", "results": []}

        results = []
        for i, doc in enumerate(relevant_docs):
            results.append({
                "document_index": i + 1,
                "page_content_preview": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content,
                "metadata": doc.metadata
            })
        
        print(f"DEBUG: Se encontraron {len(relevant_docs)} documentos relevantes.")
        return {"message": "Documentos relevantes encontrados.", "results": results}

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error al consultar el índice de Pinecone: {e}")  

class CreateIndexRequest(BaseModel):
    index_name: str
    dimension: int = 1024 # Valor por defecto, pero podrías hacerlo variable
    metric: str = "cosine"

@app.post("/create_pinecone_index")
async def create_pinecone_index(
    request_data: CreateIndexRequest
):
    try:
        # Ahora accedes a los datos a través del objeto request_data
        index_name = request_data.index_name
        dimension = request_data.dimension
        metric = request_data.metric

        pinecone_client = pinecone_utils.get_pinecone_client()

        existing_indexes = pinecone_client.list_indexes()
        print(f"DEBUG: Índices existentes en Pinecone: {[idx.name for idx in existing_indexes]}")

        if index_name in [idx.name for idx in existing_indexes]:
            print(f"DEBUG: El índice '{index_name}' ya existe.")
            return {"status": "success", "message": f"El índice '{index_name}' ya existe y está activo."}

        print(f"DEBUG: Intentando crear índice '{index_name}' con dimension {dimension} y metric {metric}...")
        spec_to_use = pinecone_utils.get_pinecone_spec()
        print(f"DEBUG: Spec de Pinecone a usar: {spec_to_use}")

        pinecone_client.create_index(
            name=index_name,
            dimension=dimension,
            metric=metric,
            spec=spec_to_use
        )
        print(f"DEBUG: ¡Índice '{index_name}' creado con éxito en Pinecone!")
        return {"status": "success", "message": f"Índice '{index_name}' creado con éxito."}
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error al crear/verificar el índice de Pinecone: {e}")
