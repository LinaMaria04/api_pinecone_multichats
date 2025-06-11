# python_api/config.py
import os #Accede a las vistas de variables de entorno
from dotenv import load_dotenv #Importa la función para cargar los datos del .env

load_dotenv() #Carga los datos del .env

#Asignación de variables de entorno a constantes en python
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY") #Busca la clave de la API de Pinecone en el .env
PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT") #Busca el entorno de Pinecone en el .env
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") #Busca la clave de la API de OpenAI en el .env
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY") #Busca la clave de la API de Anthropic en el .env

#Configración por defecto del indice de Pinecone
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "my-chatbot-index")
PINECONE_DIMENSION = int(os.getenv("PINECONE_DIMENSION", 1536)) #Convierte el valor de la variable de entorno que es un string a un numero entero
PINECONE_METRIC = os.getenv("PINECONE_METRIC", "cosine")
PINECONE_POD_TYPE = os.getenv("PINECONE_POD_TYPE", "p1.x1") # Tipo de pod para Pinecone (ajusta si tienes uno diferente)

# Modelo de embeddings de OpenAI
OPENAI_EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-ada-002")
# Modelo de chat de OpenAI
OPENAI_CHAT_MODEL = os.getenv("OPENAI_CHAT_MODEL", "gpt-3.5-turbo")