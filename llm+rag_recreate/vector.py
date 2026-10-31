# %% Import Relevant Libraries
#=====================================
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from collections import Counter
import hashlib
import json
import os
import shutil as shu
import pandas as pd
import pypdf

# %% Initial configurations
#=====================================
base_dir = os.path.dirname(os.path.abspath(__file__))
pdf_path = os.path.join(base_dir, 'pdfdir')
txt_path = os.path.join(base_dir, 'txtdir')
csv_path = os.path.join(base_dir, 'csvdir')
db_location = os.path.join(base_dir, 'dave_db')
img_path = os.path.join(base_dir, 'imgdir')
embedding_model = 'granite-embedding'
chunk_size = 1200
chunk_overlap = 240
index_version = 3
manifest_path = os.path.join(db_location, 'ingest_manifest.json')

# %% remove existing vector store if it exists - to ensure we start fresh each time
#=====================================
def remove_database():
    if os.path.exists(db_location):
        shu.rmtree(db_location)


# %% Function to process PDF files and extract text
#=====================================
def load_pdf_pages(file_path: str) -> list[Document]:
    reader = pypdf.PdfReader(file_path)
    documents = []

    for page_number, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        if text.strip():
            documents.append(
                Document(
                    page_content=text,
                    metadata={
                        "source": os.path.relpath(file_path, base_dir),
                        "page": page_number,
                    },
                )
            )

    return documents


# %% Function to convert CSV rows into Documents
#=====================================
def load_csv_documents(csv_file: str) -> list[Document]:
    documents = []

    df = pd.read_csv(csv_file)
    for i, row in df.iterrows():
        if 'Date Changed' in df.columns and 'Rate' in df.columns:
            text = f"Date Changed: {row['Date Changed']}; Rate: {row['Rate']}"
            documents.append(
                Document(
                    page_content=text,
                    metadata={
                        'source': os.path.relpath(csv_file, base_dir),
                        'row': str(i),
                        'date': row['Date Changed'],
                        'rate': row['Rate'],
                    },
                )
            )
    return documents


def get_source_files() -> list[str]:
    files = []
    for folder, extensions in ((pdf_path, {'.pdf'}), (csv_path, {'.csv'})):
        for entry in os.scandir(folder):
            if entry.is_file() and os.path.splitext(entry.name)[1].lower() in extensions:
                files.append(entry.path)
    return sorted(files)


def get_source_fingerprint(files: list[str]) -> str:
    file_state = []
    for file_path in files:
        stat = os.stat(file_path)
        file_state.append(
            {
                'path': os.path.relpath(file_path, base_dir).replace(os.sep, '/'),
                'size': stat.st_size,
                'modified_ns': stat.st_mtime_ns,
            }
        )

    state = {
        'index_version': index_version,
        'embedding_model': embedding_model,
        'chunk_size': chunk_size,
        'chunk_overlap': chunk_overlap,
        'files': file_state,
    }
    return hashlib.sha256(
        json.dumps(state, sort_keys=True).encode('utf-8')
    ).hexdigest()


def read_manifest() -> dict | None:
    try:
        with open(manifest_path, encoding='utf-8') as manifest_file:
            return json.load(manifest_file)
    except FileNotFoundError:
        return None
    except json.JSONDecodeError:
        return None


def write_manifest(fingerprint: str, document_count: int) -> None:
    os.makedirs(db_location, exist_ok=True)
    temporary_path = manifest_path + '.tmp'
    with open(temporary_path, 'w', encoding='utf-8') as manifest_file:
        json.dump(
            {'fingerprint': fingerprint, 'document_count': document_count},
            manifest_file,
            indent=2,
        )
    os.replace(temporary_path, manifest_path)


# %% Bring in embeddings and open the vector store
#=====================================
embeddings = OllamaEmbeddings(model=embedding_model)
vector_store = Chroma(
    collection_name='dave_collection',
    embedding_function=embeddings,
    persist_directory=db_location,
)

source_files = get_source_files()
source_fingerprint = get_source_fingerprint(source_files)
manifest = read_manifest()
stored_document_count = vector_store._collection.count()

if (
    manifest is not None
    and manifest.get('fingerprint') == source_fingerprint
    and manifest.get('document_count') == stored_document_count
):
    print(f'Vector store is current ({stored_document_count} chunks); source files were not reloaded.')
else:
    # %% Load source documents only when the index is missing or out of date
    #=====================================
    raw_documents = []
    for file_path in source_files:
        extension = os.path.splitext(file_path)[1].lower()
        if extension == '.pdf':
            print(f'processing pdf {os.path.basename(file_path)}')
            raw_documents.extend(load_pdf_pages(file_path))
        elif extension == '.csv':
            raw_documents.extend(load_csv_documents(file_path))

    # %% Split documents and preserve useful chunk location metadata
    #=====================================
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        add_start_index=True,
    )
    split_documents = text_splitter.split_documents(raw_documents)
    chunk_totals = Counter(
        (
            document.metadata.get('source'),
            document.metadata.get('page'),
            document.metadata.get('row'),
        )
        for document in split_documents
    )
    chunk_indices = Counter()
    for document in split_documents:
        location = (
            document.metadata.get('source'),
            document.metadata.get('page'),
            document.metadata.get('row'),
        )
        document.metadata.update(
            {
                'chunk_index': chunk_indices[location],
                'chunk_count': chunk_totals[location],
                'chunk_start': document.metadata.get('start_index', 0),
                'chunk_chars': len(document.page_content),
            }
        )
        chunk_indices[location] += 1

    print(f'Prepared {len(split_documents)} chunks from {len(raw_documents)} source documents.')

    # Recreate the derived collection so removed metadata cannot survive an upsert.
    vector_store.reset_collection()
    document_ids = [str(i) for i in range(len(split_documents))]
    batch_size = 16
    try:
        for start in range(0, len(split_documents), batch_size):
            batch_documents = split_documents[start:start + batch_size]
            vector_store.add_documents(
                documents=batch_documents,
                ids=document_ids[start:start + batch_size],
            )
    except Exception as exc:
        raise RuntimeError(
            'Unable to build the vector database. Make sure the Ollama embedding API is reachable '
            'and the granite-embedding model is installed.'
        ) from exc

    if vector_store._collection.count() != len(split_documents):
        raise RuntimeError('The vector store count does not match the number of indexed chunks.')

    write_manifest(source_fingerprint, len(split_documents))
    print(f'Indexed {len(split_documents)} chunks.')

# %% Make Vector Store reusable by the LLM
#=====================================
retriever = vector_store.as_retriever(
    search_type='similarity',
    search_kwargs={'k': 6},
)