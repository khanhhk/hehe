import glob
import multiprocessing
import pickle
from io import BytesIO
from typing import List, Union

from docling.backend.pypdfium2_backend import PyPdfiumDocumentBackend
from docling.chunking import HybridChunker
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions, TableFormerMode
from docling.document_converter import DocumentConverter, PdfFormatOption
from langchain.schema import Document
from langchain_docling.loader import DoclingLoader, ExportType
from langchain_text_splitters import RecursiveCharacterTextSplitter
from tqdm import tqdm

from ingest_data.plugins.config.minio_config import (
    MINIO_ACCESS_KEY,
    MINIO_ENDPOINT,
    MINIO_SECRET_KEY,
)
from ingest_data.plugins.jobs.utils import MinioLoader, get_tokenizer

minio_loader = MinioLoader(MINIO_ENDPOINT, MINIO_ACCESS_KEY, MINIO_SECRET_KEY)


def get_num_cpu() -> int:
    """Get number of available CPU cores."""
    return multiprocessing.cpu_count()


def create_advanced_converter() -> DocumentConverter:
    """
    Create a Docling DocumentConverter that supports both PDF and DOCX parsing.

    Returns:
        DocumentConverter: Configured for table structure extraction in PDFs.
    """
    pdf_pipeline_options = PdfPipelineOptions()
    pdf_pipeline_options.do_ocr = False
    pdf_pipeline_options.do_table_structure = True  # Bật nhận dạng cấu trúc bảng
    pdf_pipeline_options.table_structure_options.do_cell_matching = True
    # Sử dụng chế độ chính xác cao nhất để nhận dạng bảng
    pdf_pipeline_options.table_structure_options.mode = TableFormerMode.ACCURATE

    # Tạo converter hỗ trợ cả PDF và DOCX
    doc_converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(
                pipeline_options=pdf_pipeline_options, backend=PyPdfiumDocumentBackend
            ),
            # Support cho DOCX - Docling tự động xử lý DOCX tốt với default settings
            InputFormat.DOCX: None,
        }
    )

    print("-> Converter đã được cấu hình cho PDF và DOCX")
    return doc_converter


class LoadAndChunk:
    """
    Document processing pipeline for reading, chunking, and storing PDF/DOCX files.

    Features:
    - Multi-format support (PDF, DOCX, DOC)
    - Token-based recursive chunking
    - Upload/download to MinIO
    """

    def __init__(
        self,
        embed_model_id: str = (
            "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
        ),
        max_tokens: int = 512,
        chunk_overlap: int = 50,
    ) -> None:
        """
        Initialize LoadAndChunk with tokenizer and split settings.

        Args:
            embed_model_id: HuggingFace embedding model ID.
            max_tokens: Max number of tokens per chunk.
            chunk_overlap: Overlapping tokens between adjacent chunks.
        """
        self.embed_model_id = embed_model_id
        self.max_tokens = max_tokens
        self.chunk_overlap = chunk_overlap
        self.num_processes = get_num_cpu()

        # Lazy initialization
        self.converter = None
        self.tokenizer = None
        self.recursive_splitter = None

    def _init_converter(self) -> None:
        """Lazily initialize the document converter."""
        if self.converter is None:
            self.converter = create_advanced_converter()

    def _init_tokenizer_and_splitter(self) -> None:
        """Lazily initialize tokenizer and recursive splitter."""
        if self.tokenizer is None:
            print(f"-> Initializing tokenizer: {self.embed_model_id}")
            self.tokenizer = get_tokenizer()

            self.recursive_splitter = (
                RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
                    tokenizer=self.tokenizer,
                    chunk_size=self.max_tokens,
                    chunk_overlap=self.chunk_overlap,
                )
            )

    def read_and_chunk(self, files: Union[str, List[str]]) -> List[Document]:
        """
        Load and chunk one or more document files using Docling and LangChain.

        Args:
            files: File path or list of file paths.

        Returns:
            List[Document]: Tokenized and split document chunks.
        """
        if isinstance(files, str):
            files = [files]

        # Khởi tạo converter và tokenizer
        self._init_converter()
        self._init_tokenizer_and_splitter()

        assert self.tokenizer is not None
        assert self.recursive_splitter is not None

        # Filter supported formats
        supported_files = [
            f for f in files if f.lower().endswith((".pdf", ".docx", ".doc"))
        ]
        if not supported_files:
            raise ValueError("No supported files found. Accepted: PDF, DOCX, DOC.")

        if len(supported_files) != len(files):
            unsupported = [f for f in files if f not in supported_files]
            print(f"Warning: Skipping unsupported files: {unsupported}")

        print(f"Processing {len(supported_files)} file(s)...")

        all_docs = []

        for file_path in tqdm(supported_files, desc="Processing files", unit="file"):
            print(f"\n-> Bắt đầu đọc và chunking tài liệu: {file_path}")

            # Khởi tạo DoclingLoader với converter đã tùy chỉnh
            loader = DoclingLoader(
                file_path=[file_path],  # DoclingLoader expects a list
                export_type=ExportType.DOC_CHUNKS,
                converter=self.converter,
                chunker=HybridChunker(tokenizer=self.embed_model_id),
            )

            # Lấy các chunk ban đầu từ Docling
            initial_docs = loader.load()
            print(f"==> Số chunk ban đầu: {len(initial_docs)}")

            # Xử lý hậu kỳ để đảm bảo các chunk không vượt quá max_tokens
            print(
                f"-> Bắt đầu xử lý hậu kỳ để đảm bảo các chunk không vượt quá "
                f"{self.max_tokens} token..."
            )

            final_splits = []
            oversized_chunks_count = 0

            for doc in initial_docs:
                # Đếm số token trong chunk hiện tại
                num_tokens = len(
                    self.tokenizer.encode(doc.page_content, add_special_tokens=False)
                )

                if num_tokens > self.max_tokens:
                    oversized_chunks_count += 1
                    # Nếu chunk quá lớn, dùng recursive_splitter để chia nhỏ nó ra
                    sub_splits = self.recursive_splitter.split_documents([doc])
                    final_splits.extend(sub_splits)
                else:
                    # Nếu chunk có kích thước ổn, giữ nguyên nó
                    final_splits.append(doc)

            print(f"==> {oversized_chunks_count} oversized chunk(s) were split.")
            print(f"==> Final total chunks: {len(final_splits)}")

            all_docs.extend(final_splits)

        return all_docs

    def ingest_to_minio(self, data: List[Document], s3_path: str) -> None:
        """
        Serialize and upload document chunks to MinIO.

        Args:
            data: List of chunked documents.
            s3_path: Destination path in MinIO (bucket/key).
        """
        print(f"-> Uploading serialized data to: {s3_path}")
        buffer = BytesIO()
        pickle.dump(data, buffer)
        data_length = buffer.tell()
        buffer.seek(0)  # Đưa con trỏ về đầu để MinioLoader có thể đọc

        minio_loader.upload_object_from_stream(
            s3_path=s3_path, data_stream=buffer, data_length=data_length
        )
        print("-> Ingest thành công!")

    def load_from_minio(self, s3_path: str) -> List[Document]:
        """
        Load and deserialize document chunks from MinIO.

        Args:
            s3_path: Path in MinIO.

        Returns:
            List[Document]: Deserialized list of documents.
        """
        print(f"-> Downloading and deserializing from: {s3_path}")
        buffer = minio_loader.download_object_as_stream(s3_path)
        data = pickle.load(buffer)
        print("-> Load successful.")
        return data

    def load_dir(self, dir_path: str) -> List[str]:
        """
        Scan a directory for PDF and Word documents.

        Args:
            dir_path: Folder path to scan.

        Returns:
            List[str]: List of document file paths.
        """
        # Support both PDF and Word files
        pdf_files = glob.glob(f"{dir_path}/*.pdf")
        word_files = glob.glob(f"{dir_path}/*.docx") + glob.glob(f"{dir_path}/*.doc")

        all_files = pdf_files + word_files

        if not all_files:
            raise ValueError(f"No PDF or Word document files found in {dir_path}")

        print(f"Found {len(pdf_files)} PDFs and {len(word_files)} Word files.")
        return all_files

    def process_directory(self, dir_path: str) -> List[Document]:
        """
        Process all documents in a directory.

        Args:
            dir_path: Directory containing PDF/DOCX/DOC files.

        Returns:
            List[Document]: List of fully chunked documents.
        """
        files = self.load_dir(dir_path)
        return self.read_and_chunk(files)
