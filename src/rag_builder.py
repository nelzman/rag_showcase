import os
import getpass
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.callbacks import get_openai_callback
from langchain import hub
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.runnables import Runnable

class RAGBuilder:

    def __init__(self):
        pass	
    
    def make_chain_from_pdf(self, pdf_path: str) -> Runnable:

        loader = PyPDFLoader(pdf_path)
        docs = loader.load_and_split()
        # print(pages[2])
        os.environ['OPENAI_API_KEY'] = getpass.getpass('OpenAI API Key:')
            
        
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
        splits = text_splitter.split_documents(docs)
        vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings())

        # Retrieve and generate using the relevant snippets of the blog.
        retriever = vectorstore.as_retriever()
        prompt = hub.pull("rlm/rag-prompt")
        llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)


        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)


        rag_chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )
        return rag_chain
    
    
if __name__ == "__main__":
    rag_builder = RAGBuilder()
    rag_chain = rag_builder.make_chain_from_pdf(pdf_path="./docs/Baerengeschichte.pdf")
    with get_openai_callback() as cb:
        result = rag_chain.invoke("Ging Benny den Bach entlang oder hat er ihn überquert?")
        print(result)
        print(cb)