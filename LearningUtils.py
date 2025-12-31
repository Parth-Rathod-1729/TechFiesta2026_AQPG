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

def openAI_LLM7(apiKey,temperature = 0.7):
    from langchain_openai import ChatOpenAI
    return ChatOpenAI(
        base_url="https://api.llm7.io/v1",
        api_key=apiKey,
        model="gpt-4o-mini",
        temperature=temperature
    )

def huggingFaceEmbedding(hfToken = None):
    import os
    from dotenv import load_dotenv
    from langchain_huggingface import HuggingFaceEndpointEmbeddings
    load_dotenv()
    HF_TOKEN = os.getenv("HUGGINGFACE_HF_TOKEN")
    return HuggingFaceEndpointEmbeddings(
        huggingfacehub_api_token = HF_TOKEN,
        model="sentence-transformers/all-MiniLM-L6-v2",
        task="feature-extraction"
    )

def getVectorStoreOf(textPath: str, DBpath: str, chunkSize: int = 1500, chunkOverlap: int = None, chunkOverlapFraction: float = 0.2):
    import os.path
    from langchain_chroma import Chroma
    from langchain_community.document_loaders import PyPDFLoader
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    import LearningUtils
    if not os.path.exists(DBpath):
        textFile = PyPDFLoader(textPath)
        textPages = textFile.load()
        textFullContent = " ".join([txtPage.page_content.replace("\n", " ") for txtPage in textPages])
        textSplitter = RecursiveCharacterTextSplitter(
            chunk_size = chunkSize,
            chunk_overlap = chunkOverlap if chunkOverlap else int(chunkSize*chunkOverlapFraction)
        )
        textChunks = textSplitter.create_documents([textFullContent])
        return Chroma.from_documents(
            documents=textChunks,
            embedding=LearningUtils.huggingFaceEmbedding(),
            persist_directory=DBpath
        )
    else:
        return Chroma(
            embedding_function=LearningUtils.huggingFaceEmbedding(),
            persist_directory=DBpath
        )
