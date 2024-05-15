import os
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAI
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain
from PyPDF2 import PdfReader
from langchain_community.document_loaders import PyPDFLoader
from langchain.memory import ConversationBufferMemory
from langchain_community.document_loaders import DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# from langchain.embeddings import HuggingFaceInstructEmbeddings


from langchain.text_splitter import CharacterTextSplitter

from langchain.chains.question_answering import load_qa_chain
# from langchain_community.llms import OpenAI

client = OpenAI()

vectorstore = None
conversation_chain = None
chat_history = []



# proccesses a single pdf
# call this function for each pdf that needs proccessing 
def proccess_pdf(file_name):
    text = "" 

    pdf_reader = PdfReader(file_name)
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

#divides raw text into smaller chucks to be embedded 
#improves response time as llm doesnt have to sort through all the content on the page but rather just chunks
def get_text_chunks(raw_text):
    text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=20,
    length_function=len,
    is_separator_regex=False,
    )
    chunks = text_splitter.split_text(raw_text)
    return chunks

#keeps the conversational chain going and defines the llm to provide responses
def get_conversation_chain(vectorstore):
    llm = ChatOpenAI()
    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain


def main():
    global chat_history,vectorstore,conversation_chain

    #list of pdfs
    #can be imporved to request a set of PDFs before the chatbot is initialized 
    raw_text = ""
    # loader = DirectoryLoader("./pdfs/", glob="**/*pdf")
    # documents = loader.load()
    # for i in range(len(documents)):
    #     raw_text += proccess_pdf(documents[i])
    file1 = "pdfs/Yosemite_data.pdf"
    raw_text += proccess_pdf(file1)

    #split raw_text into chunks to be embeded
    text_chunks = get_text_chunks(raw_text)

    print(len(text_chunks))
    print(len(raw_text))
    # print(raw_text)
    # for i in range(len(text_chunks)):
    #     print(text_chunks[i])

    embeddings = OpenAIEmbeddings()
    # #embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")

    #storing text_chunks into a vector 
    vectorstore = FAISS.from_texts(text_chunks,embedding=embeddings)

    while True:
        #taking the user input
        querry = input("User: ")
        if querry.lower() in ["quit","bye","goodbye"]:
            break
        relevent_docs = vectorstore.similarity_search_with_score(query=querry)
        chain = get_conversation_chain(vectorstore)
        #invoked the llm, already passed the vectors earlier
        #preforms search for vector and synthasizes it 
        response = chain.invoke(querry)
        chat_history = response['chat_history']
        print("chatbot: ", response['answer'])
        


if __name__ == '__main__':
    main()
