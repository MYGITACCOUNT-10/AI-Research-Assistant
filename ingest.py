import os
import re
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma


# 1. LOAD PDF DOCUMENTS

DATA_DIR = r"C:\Users\Hitesh Kumar Patel\Desktop\AI-Research Assistant\Papers"

loader = DirectoryLoader(
    path=DATA_DIR,
    glob="**/*.pdf",
    loader_cls=PyPDFLoader,
    show_progress=True,
    use_multithreading=True
)

documents = loader.load()
print(f"Loaded {len(documents)} pages.")

# Add clean paper name metadata
for doc in documents:
    doc.metadata["paper_name"] = os.path.basename(doc.metadata.get("source", ""))


# 2. SPLIT INTO CHUNKS

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

chunks = splitter.split_documents(documents)
print(f"Total chunks before filtering: {len(chunks)}")


# 3. FILTER BAD CHUNKS

def is_bad_chunk(text: str) -> bool:
    text_lower = text.lower()

    reference_keywords = [
        "references",
        "bibliography",
        "acknowledgements",
        "acknowledgments",
        "copyright",
        "conflict of interest"
    ]

    if any(k in text_lower for k in reference_keywords):
        return True

    if text_lower.count("http") + text_lower.count("doi") >= 2:
        return True

    if len(re.findall(r"\[\d+\]", text)) >= 3:
        return True

    if len(text.strip()) < 200:
        return True

    return False


filtered_chunks = [c for c in chunks if not is_bad_chunk(c.page_content)]

print(f"Chunks after filtering: {len(filtered_chunks)}")


# 4. CREATE EMBEDDINGS & STORE

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

CHROMA_DIR = "./chroma_db"
research_vector_db = Chroma.from_documents(
    documents=filtered_chunks,
    embedding=embedding_model,
    persist_directory=CHROMA_DIR,
    collection_name="research_papers"
)

print("Chroma DB created successfully.")


