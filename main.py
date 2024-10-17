# Importamos las librerías necesarias
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, StorageContext
from llama_index.vector_stores.pinecone import PineconeVectorStore
from llama_index.llms.openai import OpenAI
import pinecone
import openai

# Definimos la clase para el cuerpo del request que el usuario enviará
class UserRequest(BaseModel):
    message: str

# Configuramos FastAPI
app = FastAPI()

# Definimos variables de entorno para acceder a las API keys de Pinecone y OpenAI
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
pc = pinecone.Pinecone(api_key=PINECONE_API_KEY)

# Definimos el nombre del índice de Pinecone
INDEX_NAME = "preguntas-frecuentes"

# Inicializamos Pinecone y verificamos que el índice exista
if INDEX_NAME not in [index.name for index in pc.list_indexes()]:
    # Si el índice no existe, lo creamos
    pc.create_index(name=INDEX_NAME, dimension=1536, metric="cosine", spec=pinecone.ServerlessSpec(cloud='aws', region='us-east-1'))

# Conectamos al índice Pinecone
pinecone_index = pc.Index(INDEX_NAME)

# Creamos un almacén vectorial utilizando Pinecone
vector_store = PineconeVectorStore(pinecone_index)

# Definimos el contexto de almacenamiento que se usará por LlamaIndex
storage_context = StorageContext.from_defaults(vector_store=vector_store)

# Cargamos los documentos desde un directorio local para convertirlos en embeddings
# Nota: Cambia el directorio "datastore" por la ubicación donde se encuentran tus datos.
documents = SimpleDirectoryReader("data").load_data()

# Creamos el índice VectorStoreIndex a partir de los documentos cargados
vector_store_index = VectorStoreIndex.from_documents(documents, storage_context=storage_context)

# Inicializamos un motor de consulta a partir del índice creado con nuestro System Prompt
system_prompt = """
Eres un chatbot de la agencia de gobierno peruano Promperu y tu objetivo es ayudar a los usuarios a responder preguntas frecuentes sobre exportación de Perú al extranjero. 

Responde siempre en español, de manera clara y concisa.
"""

llm = OpenAI(model="gpt-4o-mini")
#query_engine = vector_store_index.as_query_engine(system_prompt=system_prompt, llm_model_name="gpt-4o-mini", language="es")
chat_engine = vector_store_index.as_chat_engine(system_prompt=system_prompt, llm=llm)


# Definimos el endpoint de la API para recibir solicitudes del chatbot
@app.post("/chat")
async def chat_with_bot(request: UserRequest):
    try:
        # Recuperamos el mensaje enviado por el usuario
        user_message = request.message

        # Usamos el motor de consulta para obtener la respuesta del bot
        response = chat_engine.query(user_message)

        # Devolvemos la respuesta en formato JSON
        print(f"Response total: {response}")
        return {"response": response.response}
    except Exception as e:
        # En caso de error, devolvemos un mensaje de error con el detalle del mismo
        raise HTTPException(status_code=500, detail=str(e))

# Explicación de cada parte del código:
# 1. Importamos las librerías necesarias para construir el chatbot (FastAPI, Pydantic, Pinecone, LlamaIndex, etc.).
# 2. Definimos las variables de entorno que contienen las claves API necesarias para Pinecone y OpenAI.
# 3. Inicializamos Pinecone y verificamos que el índice que necesitamos esté disponible.
# 4. Si el índice no está disponible, lo creamos.
# 5. Cargamos los documentos locales y creamos el índice VectorStoreIndex con LlamaIndex para almacenar los embeddings en Pinecone.
# 6. Definimos un endpoint "/chat" utilizando FastAPI, que permite al usuario enviar un mensaje y recibir una respuesta del bot en un POST request.

# Ejecución: Este script se debe ejecutar con uvicorn para correr el servidor de FastAPI. Por ejemplo:
# uvicorn main:app --reload

# Asegúrate de instalar las dependencias necesarias ejecutando los siguientes comandos:
# pip install fastapi uvicorn llama-index pinecone-client openai llama-index-vector-stores-pinecone

# Cambia "data" a la ruta donde tienes los documentos que quieres utilizar para el contexto del bot.
