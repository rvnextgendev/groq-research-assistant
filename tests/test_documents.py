from io import BytesIO
from unittest.mock import patch

import pytest
from langchain_core.embeddings import DeterministicFakeEmbedding
from pypdf import PdfWriter
from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject

from research_assistant.documents import build_index, parse_pdf
from research_assistant.errors import DocumentError


def sample_pdf(text=True):
    writer = PdfWriter()
    page = writer.add_blank_page(width=300, height=300)
    if text:
        font = DictionaryObject({NameObject("/Type"): NameObject("/Font"),
            NameObject("/Subtype"): NameObject("/Type1"), NameObject("/BaseFont"): NameObject("/Helvetica")})
        page[NameObject("/Resources")] = DictionaryObject({NameObject("/Font"):
            DictionaryObject({NameObject("/F1"): writer._add_object(font)})})
        stream = DecodedStreamObject()
        stream.set_data(b"BT /F1 12 Tf 10 100 Td (Revenue grew 34 percent.) Tj ET")
        page[NameObject("/Contents")] = writer._add_object(stream)
    output = BytesIO()
    writer.write(output)
    return output.getvalue()


@pytest.mark.parametrize("data", [b"", b"not a pdf", b"%PDF-broken"])
def test_invalid_pdf_rejected(settings, data):
    with pytest.raises(DocumentError):
        parse_pdf(data, settings)


def test_scanned_pdf_rejected(settings):
    with pytest.raises(DocumentError, match="OCR"):
        parse_pdf(sample_pdf(False), settings)


def test_index_retrieves_page_citation(settings):
    index = build_index(sample_pdf(), settings, DeterministicFakeEmbedding(size=16))
    assert index.page_count == 1
    assert index.chunk_count == 1
    result = index.retriever.invoke("revenue")[0]
    assert result.metadata["page_number"] == 1
    assert "34 percent" in result.page_content


def test_upload_limit_precedes_parsing(settings):
    with patch("research_assistant.documents.PdfReader") as reader:
        with pytest.raises(DocumentError):
            parse_pdf(b"%PDF-" + b"x" * (settings.max_upload_mb * 1024 * 1024), settings)
        reader.assert_not_called()
