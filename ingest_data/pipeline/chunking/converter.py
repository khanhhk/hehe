from docling.backend.pypdfium2_backend import PyPdfiumDocumentBackend
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions, TableFormerMode
from docling.document_converter import DocumentConverter, PdfFormatOption


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
