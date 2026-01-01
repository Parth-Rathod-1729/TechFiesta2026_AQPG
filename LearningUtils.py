def rateLimiter(waitSec: int = 2, showOutput: bool = True):
    import time
    print(f"\n\tWaiting {waitSec} sec", end = "")
    for _ in range(waitSec):
        time.sleep(1)
        print(".", end="") if showOutput else None
    print()

def gemini_2_5_flash(temperature: float = 0.3):
    from langchain_google_genai import ChatGoogleGenerativeAI
    return ChatGoogleGenerativeAI(
        model = "gemini-2.5-flash",
        temperature = temperature
    )

def openAI_LLM7(temperature = 0.7, apiKey = None):
    from langchain_openai import ChatOpenAI
    import os
    from dotenv import load_dotenv
    load_dotenv()
    LLM7_TOKEN = os.getenv("LLM7_TOKEN")
    return ChatOpenAI(
        base_url="https://api.llm7.io/v1",
        api_key=LLM7_TOKEN or apiKey,
        model="gpt-4o-mini",
        temperature=temperature
    )

def huggingFaceEmbedding(hfToken = None, timeoutSec: int = 600):
    import os
    from dotenv import load_dotenv
    from langchain_huggingface import HuggingFaceEndpointEmbeddings
    load_dotenv()
    HF_TOKEN = os.getenv("HUGGINGFACE_HF_TOKEN")
    class TimeoutWrapperClass_for_HuggingFaceEndpointEmbeddings(HuggingFaceEndpointEmbeddings):
        def __init__(self, timeout: int = 600, **kwargs):
            super().__init__(**kwargs)
            self.client.timeout = timeout

    return TimeoutWrapperClass_for_HuggingFaceEndpointEmbeddings(
        timeout = timeoutSec,
        huggingfacehub_api_token = HF_TOKEN,
        model="sentence-transformers/all-MiniLM-L6-v2",
        task="feature-extraction"
    )

def getVectorStoreOf(textPDFpath: str, databasePath: str, chunkSize: int = 1500, chunkOverlap: int = None, chunkOverlapFraction: float = 0.2, timeoutSec: int = 600):
    import os.path
    from langchain_chroma import Chroma
    from langchain_community.document_loaders import PyPDFLoader
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    import LearningUtils
    if not os.path.exists(databasePath):
        textFile = PyPDFLoader(textPDFpath)
        textPages = textFile.load()
        textFullContent = " ".join([txtPage.page_content.replace("\n", " ") for txtPage in textPages])
        textSplitter = RecursiveCharacterTextSplitter(
            chunk_size = chunkSize,
            chunk_overlap = chunkOverlap if chunkOverlap else int(chunkSize*chunkOverlapFraction)
        )
        textChunks = textSplitter.create_documents([textFullContent])
        return Chroma.from_documents(
            documents=textChunks,
            embedding=LearningUtils.huggingFaceEmbedding(timeoutSec=timeoutSec),
            persist_directory=databasePath
        )
    else:
        return Chroma(
            embedding_function=LearningUtils.huggingFaceEmbedding(timeoutSec=timeoutSec),
            persist_directory=databasePath
        )
