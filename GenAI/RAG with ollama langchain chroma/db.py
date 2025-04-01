from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import MarkdownHeaderTextSplitter

loader = TextLoader('RAG with ollama langchain chroma/information.md')
document = loader.load()[0]

# print(document)

headers_to_split_on = [
    ("#", "Header 1"),
    ("##", "Header 2"), 
    ("###", "Header 3")
]

splitter = MarkdownHeaderTextSplitter(
    headers_to_split_on=headers_to_split_on,
    strip_headers=False 
)

chunks = splitter.split_text(document.page_content)
print("\n\nDone\n\n")

# print(markdown_splitter)

print(f"Total chunks: {len(chunks)}")
for i, chunk in enumerate(chunks): 
    print(f"\nChunk {i+1}:")
    print(f"Metadata: {chunk.metadata}")
    print(f"Content preview: {chunk.page_content}")
    
embeddings = OllamaEmbeddings(model='mxbai-embed-large')

db_location = './RAG with ollama langchain chroma/chroma_db'


vector_store = Chroma(
    collection_name="space_information",
    embedding_function=embeddings,
    persist_directory=db_location
)

documents = []
ids = []

for i, chunk in enumerate(chunks):
    document = Document(
        page_content=chunk.page_content,
        metadata=chunk.metadata,
        id=str(i+1)
    )
    documents.append(document)
    ids.append(str(i+1))

vector_store.add_documents(documents=documents,ids=ids)

retriever = vector_store.as_retriever(
    search_kwargs={'k': 2}
)

print('reached end of db.py file')