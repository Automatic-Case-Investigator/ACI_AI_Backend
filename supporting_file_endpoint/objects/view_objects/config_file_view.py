import uuid
import tempfile
import os
import logging

from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status

from ACI_AI_Backend.objects.chromadb_client import chromadb_client

logger = logging.getLogger(__name__)

COLLECTION_NAME = "config_files"
CHUNK_SIZE = 100  # words per chunk
CHUNK_OVERLAP = 50  # word overlap between consecutive chunks


def _chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Split *text* into overlapping word-based chunks.

    Parameters
    ----------
    text : str
        Source text to split.
    chunk_size : int
        Number of words per chunk.
    overlap : int
        Number of words shared between consecutive chunks.

    Returns
    -------
    list[str]
        Ordered list of text chunks.
    """
    words = text.split()
    if not words:
        return []

    chunks: list[str] = []
    start = 0
    step = chunk_size - overlap

    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunks.append(" ".join(words[start:end]))
        if end == len(words):
            break
        start += step

    return chunks


def _extract_text(uploaded_file, filename: str) -> str:
    """Extract plain text from an uploaded Django file object.

    Tries ``unstructured.partition.auto.partition`` first (handles PDF, DOCX,
    HTML, etc.).  Falls back to UTF-8 text decoding if unstructured raises.

    Parameters
    ----------
    uploaded_file : django.core.files.uploadedfile.UploadedFile
        The file received from the request.
    filename : str
        Original filename (used to derive a temp-file suffix for unstructured).

    Returns
    -------
    str
        Extracted text content.
    """
    suffix = os.path.splitext(filename)[-1].lower() or ".tmp"

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        for chunk in uploaded_file.chunks():
            tmp.write(chunk)
        tmp_path = tmp.name

    try:
        from unstructured.partition.auto import partition  # type: ignore

        elements = partition(filename=tmp_path)
        return "\n".join(str(el) for el in elements)
    except Exception:
        with open(tmp_path, "r", encoding="utf-8", errors="replace") as fh:
            return fh.read()
    finally:
        os.unlink(tmp_path)


class ConfigFileView(APIView):
    """
    Manage configuration / knowledge-base files stored in the vector database.

    POST – Upload Files
    -------------------
    Upload one or more files under the ``files`` multipart field.
    Each file is:
    1. Parsed into plain text (PDF, DOCX, TXT, … via *unstructured*).
    2. Split into overlapping chunks
    3. Stored in the ``config_files`` ChromaDB collection with
       ``{"filename": <original filename>}`` metadata.

    If a file with the same name was previously uploaded, its old chunks are
    replaced automatically before the new ones are inserted.

    Response (200):
        {
          "results": [
            {"filename": "...", "status": "success", "chunks_stored": <n>},
            ...
          ]
        }

    DELETE – Remove Files
    ---------------------
    Body (JSON): ``{"filenames": ["file1.pdf", "report.docx"]}``

    Deletes every stored chunk whose ``filename`` metadata matches one of the
    supplied names.

    Response (200):
        {
          "results": [
            {"filename": "...", "status": "deleted"},
            ...
          ]
        }
    """

    def _collection(self):
        return chromadb_client.get_or_create_collection(COLLECTION_NAME)

    # ------------------------------------------------------------------
    # POST – ingest files
    # ------------------------------------------------------------------
    def post(self, request, *args, **kwargs):
        files = request.FILES.getlist("files")
        print("Uploading files")

        if not files:
            return Response(
                {"error": "No files provided. Send files under the 'files' field."},
                status=status.HTTP_400_BAD_REQUEST,
            )

        collection = self._collection()

        for uploaded_file in files:
            filename = uploaded_file.name
            print(f"Adding file: {filename}")

            # Replace any pre-existing chunks for this filename
            try:
                print("Trying to delete old file")
                collection.delete(where={"filename": {"$eq": filename}})
            except Exception:
                print("Failed to delete old file")
                pass  # Collection may be empty; ignore

            # Extract text
            try:
                print("Extracting text of the file")
                text = _extract_text(uploaded_file, filename)
            except Exception as exc:
                logger.exception("Text extraction failed for %s", filename)
                print("Text extraction failed")
                continue

            if not text.strip():
                print("Text is empty")
                continue

            # Chunk and insert
            print("Chunking text")
            chunks = _chunk_text(text)
            print(len(chunks))
            ids = [str(uuid.uuid4()) for _ in chunks]
            metadatas = [{"filename": filename} for _ in chunks]

            print("Adding file to collection")
            batch_size = 500

            for i in range(0, len(chunks), batch_size):
                print(i)
                batch_ids = ids[i:i+batch_size]
                batch_docs = chunks[i:i+batch_size]
                batch_metadatas = metadatas[i:i+batch_size]
                collection.add(ids=batch_ids, documents=batch_docs, metadatas=batch_metadatas)
            
            print(f"Added file: {filename}")

        return Response({"message": "Success"}, status=status.HTTP_200_OK)

    # ------------------------------------------------------------------
    # DELETE – remove files by name
    # ------------------------------------------------------------------
    def delete(self, request, *args, **kwargs):
        data = request.data or {}
        filenames = data.get("filenames", [])

        if not isinstance(filenames, list) or not filenames:
            return Response(
                {"error": "Provide a non-empty 'filenames' list in the request body."},
                status=status.HTTP_400_BAD_REQUEST,
            )

        if not all(isinstance(f, str) for f in filenames):
            return Response(
                {"error": "'filenames' must be a list of strings."},
                status=status.HTTP_400_BAD_REQUEST,
            )

        collection = self._collection()
        results: list[dict] = []

        for filename in filenames:
            try:
                collection.delete(where={"filename": {"$eq": filename}})
                results.append({"filename": filename, "status": "deleted"})
            except Exception as exc:
                logger.exception("Failed to delete chunks for %s", filename)
                results.append({"filename": filename, "status": "error", "detail": str(exc)})

        return Response({"message": "Success", "results": results}, status=status.HTTP_200_OK)
