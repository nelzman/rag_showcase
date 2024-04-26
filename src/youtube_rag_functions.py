
import os

from langchain import hub
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import Runnable, RunnablePassthrough
from langchain_community.callbacks import get_openai_callback
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
import openai
from pytube import YouTube

      

def get_audio_from_youtube(url: str = 'https://youtu.be/2lAe1cqCOXo', audio_path: str = "../artifacts/audio") -> str:
    """
    :param url: string url
    :param video_path: string video_path
    
    - download video from youtube
    """
    os.makedirs(audio_path, exist_ok=True)
    audio = YouTube(url).streams.filter(only_audio=True).first()
    audio.download(output_path=audio_path)
    return audio.default_filename

def get_video_from_youtube(url: str = 'https://youtu.be/2lAe1cqCOXo', video_path: str = "../artifacts/video") -> str:
    """
    :param url: string url
    :param video_path: string video_path
    
    - download video from youtube
    """
    os.makedirs(video_path, exist_ok=True)
    video = YouTube(url).streams.first()
    video.download(output_path=video_path)
    return video.default_filename
    
def get_docs_from_youtube(client: object, url: str, audio_path: str = "../artifacts/audio") -> list:
    """
    :param url: string url
    :return: list

    - get documents from youtube
    """
    audio_file = get_audio_from_youtube(url=url, audio_path=audio_path)
    text = speech_to_text(client=client, audio_file_name=audio_file, file_path=audio_path, language="en")
    doc = Document(
                page_content=text,
                metadata={"source": audio_file, "page": 1},
            )
    return [doc]

def speech_to_text(client: object, audio_file_name: str = "speech_recording.mp4", 
                   file_path: str = "../artifacts/audio", language: str = "en") -> str:
    """
    :param audio_file_name: string audio_file_name
    :param language: string language
    :return: string

    - get a response from chatGPT
    - convert speech to text
    """
    with open(f"{file_path}/{audio_file_name}", "rb") as audio_file:
        response = client.audio.transcriptions.create(model="whisper-1", language=language, file=audio_file)
        gpt_result = response.text

    return gpt_result

def split_docs(docs: list, chunk_size: int = 1000, chunk_overlap: int = 200) -> list:
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    splits = text_splitter.split_documents(docs)

    return splits

def create_vectorstore(
    splits: list,
    vectore_store_name: str = "chromadb",
):  
    """
    :param splits: list[Document]
    :param vectore_store_name: string vectore_store_name
    :return: object

    - create a vectorstore from the pages
    """
    print("Create a vectorstore from the pages.")

    embeddings = OpenAIEmbeddings()
    if vectore_store_name == "chromadb":
        from langchain_community.vectorstores import Chroma

        vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
    elif vectore_store_name == "deeplake":
        from langchain_community.vectorstores import DeepLake

        vectorstore = DeepLake.from_documents(
            documents=splits,
            dataset_path="./my_deeplake/",
            embedding=embeddings,
            overwrite=True,
        )
    elif vectore_store_name == "faiss":
        from langchain_community.vectorstores import FAISS

        vectorstore = FAISS.from_documents(splits, embeddings)
    elif vectore_store_name == "annoy":
        from langchain_community.vectorstores import Annoy

        vectorstore = Annoy.from_documents(splits, embeddings)
    elif vectore_store_name == "docarray_hnsw_search":
        from langchain.vectorstores.docarray.hnsw import DocArrayHnswSearch

        # vectorstore = DocArrayHnswSearch.from_params(embeddings, f"tmp/airbyte_local/{boardgame}", 1536)
        vectorstore = DocArrayHnswSearch.from_documents(splits, embeddings, work_dir="hnswlib_store/", n_dim=1536)
    elif vectore_store_name == "docarray_inmemory":
        from langchain_community.vectorstores.docarray import DocArrayInMemorySearch

        vectorstore = DocArrayInMemorySearch.from_documents(splits, embeddings)
    else:
        raise ValueError("Unknown vectorstore name")

    return vectorstore

def create_retriever(vectorstore: object):
    return vectorstore.as_retriever()

def get_prompt():
    #prompt = hub.pull("rlm/rag-prompt")
    prompt = hub.pull("nelzman/rag-prompt-with-dunno-catch")
    return prompt

def create_language_model():
    llm = ChatOpenAI(
        model_name="gpt-3.5-turbo-1106",  # currently only works with gpt-3.5-turbo-1106
        temperature=0,
    )
    return llm

def create_rag_chain(retriever, prompt, llm):
    """
    :param retriever: vectorstore retreiver
    :param prompt: prompt for the question

    
    - create chain: retriever -> format_docs -> prompt -> llm -> StrOutputParser
    """

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()} | prompt | llm | StrOutputParser()
    )

    return rag_chain

def check_rag_chain(rag_chain: object, prompt: str = None):
    with get_openai_callback() as cb:
        response = rag_chain.invoke(prompt)
    return response, cb

