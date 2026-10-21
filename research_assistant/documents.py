"""Bounded in-memory PDF parsing and indexing."""
from dataclasses import dataclass
from io import BytesIO

from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

from research_assistant.config import Settings
from research_assistant.errors import DocumentError


@dataclass(frozen=True)
class DocumentIndex:
    """Retriever and display statistics belonging to one uploaded document."""
    retriever: VectorStoreRetriever
    page_count: int
    chunk_count: int


def parse_pdf(data: bytes, settings: Settings) -> list[Document]:
    """Extract page text without saving uploaded bytes to the filesystem."""
    # Reject oversized input before invoking the parser. These limits do not sandbox
    # the parser or guarantee a hard CPU/decompression bound for hostile PDFs.
    if not data or len(data) > settings.max_upload_mb * 1024 * 1024:
        raise DocumentError(f"Upload a nonempty PDF up to {settings.max_upload_mb} MB.")
    if not data.lstrip().startswith(b"%PDF-"):
        raise DocumentError("The uploaded file is not a PDF.")
    try:
        # BytesIO lets pypdf read the upload as a file without creating a temp file.
        reader = PdfReader(BytesIO(data))
        if reader.is_encrypted:
            raise DocumentError("Upload an unencrypted PDF.")
        if len(reader.pages) > settings.max_pages:
            raise DocumentError(f"PDF must have at most {settings.max_pages} pages.")
        documents, total = [], 0
        for number, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""
            total += len(text)
            if total > settings.max_text_chars:
                raise DocumentError("PDF contains too much text. Upload a smaller document.")
            # One-based physical page numbers survive splitting for source citations.
            documents.append(Document(page_content=text, metadata={"page_number": number}))
    except DocumentError:
        raise
    except Exception:
        # Parser details are not useful UI messages and can include document data.
        raise DocumentError("Could not read this PDF. Upload a valid, unencrypted file.") from None
    if not any(doc.page_content.strip() for doc in documents):
        # Text extraction does not perform OCR on image-only scans.
        raise DocumentError("No extractable text found. Run OCR on scanned PDFs first.")
    return documents


def build_index(data: bytes, settings: Settings, embeddings: Embeddings | None = None) -> DocumentIndex:
    """Split and embed the PDF; allow injected embeddings for offline tests."""
    documents = parse_pdf(data, settings)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size, chunk_overlap=settings.chunk_overlap)
    chunks = splitter.split_documents(documents)
    # The default model runs locally and downloads its assets on first use.
    embeddings = embeddings or FastEmbedEmbeddings(model_name=settings.embedding_model)
    store = FAISS.from_documents(chunks, embeddings)
    # No save_local call: the vector index remains in this session's memory.
    return DocumentIndex(store.as_retriever(search_kwargs={"k": settings.retrieval_k}),
                         len(documents), len(chunks))
