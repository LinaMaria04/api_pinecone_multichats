# python_api/main.py
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from pydantic import BaseModel
from langchain_core.documents import Document
from typing import Optional, Dict, Any, List, Union
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

#pc = os.getenv("PINECONE_API_KEY")

# Modelos Pydantic para la validación de datos de entrada
class DataToUpsert(BaseModel):
    id: str
    text_content: str
    metadata: dict[str, Any] #Metadatos adicionales para el documento
    vectorstore_id: str #ID del vectorstore (Nombre del índice en Pinecone)

class ChatRequest(BaseModel):
    question: str
    vectorstore_id: str # ID del vectorstore (Nombre del índice en Pinecone)
    comportamiento_chat: Optional[str] = "" 
    chat_history: list = [] # Campo opcional si decides enviar historial de chat
    chat_id: Optional[int] = None

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
    chat_id: Optional[Union[str, int]] = Form(None)
):
    print(f"DEBUG: chat_id recibido en ingest_file: {chat_id}")
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

        # En main.py, dentro de ingest_file, al construir base_document
        base_document_metadata = {
            "original_file_name": file.filename,
            "document_id": document_id,
            **metadata # fusionar metadatos enviados desde Laravel (que ya NO incluirá 'chat_id')
        }

        if chat_id is not None:
            base_document_metadata["chat_id"] = chat_id 

        base_document = Document(page_content=text_content, metadata=base_document_metadata)
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

        target_namespace = None
        if chat_id is not None:
            target_namespace = str(chat_id) 
            print(f"DEBUG: Ingestando en namespace: '{target_namespace}'")

        #Limpiar o añadir metadatos específicos para cada chunk
        processed_chunks = []
        for i, chunk in enumerate(chunks):
            # Copiar metadatos existentes y añadir o sobrescribir si es necesario
            new_metadata = chunk.metadata.copy()
            new_metadata["chunk_index"] = i # Índice del chunk
                       
            processed_chunks.append(Document(page_content=chunk.page_content, metadata=new_metadata))

        #Llamar a la función de ingesta de Pinecone con los CHUNKS
        print(f"DEBUG: Ingestando {len(processed_chunks)} chunks en el vectorstore '{vectorstore_id}'...")
        pinecone_utils.upsert_documents_to_pinecone(processed_chunks, vectorstore_id, name_space=chat_id)
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
            system_behavior=request.comportamiento_chat,
            chat_id=request.chat_id
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
    
class DeleteRequest(BaseModel):
    index_name: str
    document_id: str
    chat_id: Optional[Union[str, int]] = None


@app.post("/delete_from_pinecone")
async def delete_from_pinecone(request: DeleteRequest):
    try:
        pinecone_client = pinecone_utils.get_pinecone_client()
        if pinecone_client is None:
            raise HTTPException(status_code=500, detail="El cliente de Pinecone no está inicializado.")

        # Verificar el indice
        pinecone_index_descriptions = pinecone_client.list_indexes()
        existing_index_names = [idx.name for idx in pinecone_index_descriptions]

        if request.index_name not in existing_index_names:
            raise HTTPException(status_code=404, detail=f"El índice '{request.index_name}' no existe en Pinecone.")

        index = pinecone_client.Index(request.index_name)

        # Pasar el Namespace
        namespace_to_delete_from = None
        if request.chat_id is not None:
            namespace_to_delete_from = str(request.chat_id) # El namespace debe ser String
            print(f"DEBUG: Intentando eliminar en namespace: '{namespace_to_delete_from}' (Tipo: {type(namespace_to_delete_from)})")
        else:
            print("DEBUG: No se proporcionó chat_id, intentando eliminar en el namespace por defecto (cadena vacía).")

        #Filtro para eliminación
        delete_filter = {
            "document_id": request.document_id # document_id del archivo
        }

        if request.chat_id is not None:
            delete_filter["chat_id"] = str(request.chat_id) 

        print(f"DEBUG: Intentando eliminar con filtro: {delete_filter} en namespace: '{namespace_to_delete_from or 'default'}' para index '{request.index_name}'")

        #Verificar si los vectores existen ANTES de eliminarlos con el filtro
        try:
            dummy_vector = [0.0] * 384 #Dimensión del indice
            
            # Ajusta top_k para ver si encuentra múltiples chunks del mismo documento
            found_vectors = index.query(
                vector=dummy_vector, 
                top_k=10, # O un número mayor si esperas muchos chunks por archivo
                filter=delete_filter, 
                namespace=namespace_to_delete_from,
                include_metadata=True # Para ver los metadatos de los encontrados
            )
            
            if found_vectors.matches:
                print(f"DEBUG: ENCONTRADO(S) {len(found_vectors.matches)} vector(es) con el filtro ANTES DE LA ELIMINACIÓN.")
                for match in found_vectors.matches:
                    print(f"  - ID: {match.id}, Metadata: {match.metadata}")
            else:
                print(f"DEBUG: NINGÚN vector ENCONTRADO con el filtro ANTES DE LA ELIMINACIÓN. ELIMINACIÓN NO TENDRÁ EFECTO.")
        except Exception as verify_e:
            print(f"ERROR: Fallo al intentar verificar los documentos antes de eliminar: {verify_e}")

        # Eliminar de acuerdo al filtro.
        index.delete(filter=delete_filter, namespace=namespace_to_delete_from)

        return {"message": f"Vectores asociados al document_id '{request.document_id}' eliminados exitosamente del índice '{request.index_name}' en el namespace '{namespace_to_delete_from or 'default'}'."}
    except HTTPException as e:
        raise e
    except Exception as e:
        traceback.print_exc()
        print(f"Error al eliminar de Pinecone: {e}")
        raise HTTPException(status_code=500, detail=f"Error interno del servidor al eliminar de Pinecone: {str(e)}")
