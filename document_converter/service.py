import base64
import logging
from abc import ABC, abstractmethod
from io import BytesIO
from typing import List, Tuple, Optional, Dict, Any
import io
import json
import re
import os
import uuid
from datetime import datetime
from threading import Thread
from concurrent.futures import ThreadPoolExecutor, as_completed

from celery.result import AsyncResult
from docling.datamodel.base_models import InputFormat, DocumentStream
from docling.datamodel.pipeline_options import PdfPipelineOptions, EasyOcrOptions
from docling.document_converter import PdfFormatOption, DocumentConverter
from docling_core.types.doc import ImageRefMode, TableItem, PictureItem
from fastapi import HTTPException
from chonkie import SDPMChunker

from document_converter.schema import BatchConversionJobResult, ConversionJobResult, ConversionResult, ImageData, ChunkingResult, Chunk
from document_converter.utils import handle_csv_file

logging.basicConfig(level=logging.INFO)
IMAGE_RESOLUTION_SCALE = int(os.getenv("IMAGE_RESOLUTION_SCALE", "1"))


class DocumentConversionBase(ABC):
    @abstractmethod
    def convert(self, document: Tuple[str, BytesIO], **kwargs) -> ConversionResult:
        pass

    @abstractmethod
    def convert_batch(self, documents: List[Tuple[str, BytesIO]], **kwargs) -> List[ConversionResult]:
        pass


class DoclingDocumentConversion(DocumentConversionBase):
    """Document conversion implementation using Docling.

    You can initialize with default pipeline options or provide your own:

    Example:
        ```python
        # Using default options
        converter = DoclingDocumentConversion()

        # Or customize with your own pipeline options
        pipeline_options = PdfPipelineOptions()
        pipeline_options.do_ocr = True
        pipeline_options.ocr_options = RapidOcrOptions()  # Use RapidOcrOptions instead of EasyOCR (note : you need to install the OCR package)
        pipeline_options.generate_page_images = True

        converter = DoclingDocumentConversion(pipeline_options=pipeline_options)
        ```
    """

    def __init__(self, pipeline_options: PdfPipelineOptions = None):
        self.pipeline_options = pipeline_options if pipeline_options else self._setup_default_pipeline_options()

    def _update_pipeline_options(self, extract_tables: bool, image_resolution_scale: int) -> PdfPipelineOptions:
        self.pipeline_options.images_scale = image_resolution_scale
        self.pipeline_options.generate_table_images = extract_tables
        return self.pipeline_options

    @staticmethod
    def _setup_default_pipeline_options() -> PdfPipelineOptions:
        pipeline_options = PdfPipelineOptions()
        pipeline_options.generate_page_images = False
        pipeline_options.generate_picture_images = True
        pipeline_options.ocr_options = EasyOcrOptions(lang=["fr", "de", "es", "en", "it", "pt"])

        return pipeline_options

    @staticmethod
    def _process_document_images(conv_res) -> Tuple[str, List[ImageData], Optional[Dict[int, str]]]:
        images = []
        table_counter = 0
        picture_counter = 0
        content_md = conv_res.document.export_to_markdown(image_mode=ImageRefMode.PLACEHOLDER)
        
        # Extract page-by-page content
        page_content = {}
        num_pages = conv_res.document.num_pages()
        for page_num in range(1, num_pages + 1):
            page_md = conv_res.document.export_to_markdown(
                image_mode=ImageRefMode.PLACEHOLDER,
                page_no=page_num
            )
            if page_md.strip():  # Only add non-empty pages
                page_content[str(page_num)] = page_md

        for element, _level in conv_res.document.iterate_items():
            if isinstance(element, (TableItem, PictureItem)) and element.image:
                img_buffer = BytesIO()
                element.image.pil_image.save(img_buffer, format="PNG")

                if isinstance(element, TableItem):
                    table_counter += 1
                    image_name = f"table-{table_counter}.png"
                    image_type = "table"
                else:
                    picture_counter += 1
                    image_name = f"picture-{picture_counter}.png"
                    image_type = "picture"
                    content_md = content_md.replace("<!-- image -->", image_name, 1)
                    
                    # Also replace image placeholders in page content
                    for page_num, page_md in page_content.items():
                        if "<!-- image -->" in page_md:
                            page_content[page_num] = page_md.replace("<!-- image -->", image_name, 1)
                            break

                image_bytes = base64.b64encode(img_buffer.getvalue()).decode('utf-8')
                images.append(ImageData(type=image_type, filename=image_name, image=image_bytes))

        return content_md, images, page_content

    def convert(
        self,
        document: Tuple[str, BytesIO],
        extract_tables: bool = False,
        image_resolution_scale: int = IMAGE_RESOLUTION_SCALE,
    ) -> ConversionResult:
        filename, file = document
        pipeline_options = self._update_pipeline_options(extract_tables, image_resolution_scale)
        doc_converter = DocumentConverter(
            format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)}
        )

        if filename.lower().endswith('.csv'):
            file, error = handle_csv_file(file)
            if error:
                return ConversionResult(filename=filename, error=error)

        conv_res = doc_converter.convert(DocumentStream(name=filename, stream=file), raises_on_error=False)
        doc_filename = conv_res.input.file.stem

        if conv_res.errors:
            logging.error(f"Failed to convert {filename}: {conv_res.errors[0].error_message}")
            return ConversionResult(filename=doc_filename, error=conv_res.errors[0].error_message)

        content_md, images, page_content = self._process_document_images(conv_res)
        return ConversionResult(filename=doc_filename, markdown=content_md, images=images, page_content=page_content)

    def convert_batch(
        self,
        documents: List[Tuple[str, BytesIO]],
        extract_tables: bool = False,
        image_resolution_scale: int = IMAGE_RESOLUTION_SCALE,
        fallback_sequential_numbering: bool = False,
    ) -> List[ConversionResult]:
        pipeline_options = self._update_pipeline_options(extract_tables, image_resolution_scale)
        doc_converter = DocumentConverter(
            format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)}
        )

        conv_results = doc_converter.convert_all(
            [DocumentStream(name=filename, stream=file) for filename, file in documents],
            raises_on_error=False,
        )

        results = []
        for conv_res in conv_results:
            doc_filename = conv_res.input.file.stem

            if conv_res.errors:
                logging.error(f"Failed to convert {conv_res.input.name}: {conv_res.errors[0].error_message}")
                results.append(ConversionResult(filename=conv_res.input.name, error=conv_res.errors[0].error_message))
                continue

            content_md, images, page_content = self._process_document_images(conv_res)
            results.append(ConversionResult(filename=doc_filename, markdown=content_md, images=images, page_content=page_content))

        return results


class DocumentConverterService:
    def __init__(self, document_converter: DocumentConversionBase):
        self.document_converter = document_converter
        
    def _setup_default_pipeline_options(self) -> PdfPipelineOptions:
        """
        Set up default pipeline options for PDF conversion.
        
        Returns:
            PdfPipelineOptions: Default pipeline options
        """
        # Delegate to the DoclingDocumentConversion class's static method
        return DoclingDocumentConversion._setup_default_pipeline_options()

    def convert_document(self, document: Tuple[str, BytesIO], include_page_numbers: bool = False, **kwargs) -> ConversionResult:
        result = self.document_converter.convert(document, **kwargs)
        if result.error:
            logging.error(f"Failed to convert {document[0]}: {result.error}")
            raise HTTPException(status_code=500, detail=result.error)
            
        # If page numbers are requested, format the markdown with page numbers
        if include_page_numbers and result.page_content:
            result.markdown = self.get_markdown_with_page_numbers(result)
            
        return result

    def convert_documents(self, documents: List[Tuple[str, BytesIO]], include_page_numbers: bool = False, **kwargs) -> List[ConversionResult]:
        results = self.document_converter.convert_batch(documents, **kwargs)
        
        # If page numbers are requested, format the markdown with page numbers for each result
        if include_page_numbers:
            for result in results:
                if result.page_content and not result.error:
                    result.markdown = self.get_markdown_with_page_numbers(result)
                    
        return results

    def convert_document_task(
        self,
        document: Tuple[str, bytes],
        include_page_numbers: bool = False,
        **kwargs,
    ) -> ConversionResult:
        document = (document[0], BytesIO(document[1]))
        result = self.document_converter.convert(document, **kwargs)
        
        # If page numbers are requested, format the markdown with page numbers
        if include_page_numbers and result.page_content and not result.error:
            result.markdown = self.get_markdown_with_page_numbers(result)
            
        return result

    def convert_documents_task(
        self,
        documents: List[Tuple[str, bytes]],
        include_page_numbers: bool = False,
        **kwargs,
    ) -> List[ConversionResult]:
        documents = [(filename, BytesIO(file)) for filename, file in documents]
        results = self.document_converter.convert_batch(documents, **kwargs)
        
        # If page numbers are requested, format the markdown with page numbers for each result
        if include_page_numbers:
            for result in results:
                if result.page_content and not result.error:
                    result.markdown = self.get_markdown_with_page_numbers(result)
                    
        return results

    def get_single_document_task_result(self, job_id: str, include_page_numbers: bool = False) -> ConversionJobResult:
        """
        Get the result of a single document conversion task.

        Args:
            job_id: The ID of the job
            include_page_numbers: Whether to include page numbers in the markdown

        Returns:
            ConversionJobResult: The result of the conversion job
        """
        # Import celery_app only when needed to avoid circular imports
        from worker.celery_config import celery_app
        
        task = AsyncResult(job_id, app=celery_app)
        if task.state == 'PENDING':
            return ConversionJobResult(job_id=job_id, status="IN_PROGRESS")
        elif task.state == 'FAILURE':
            return ConversionJobResult(job_id=job_id, status="FAILURE", error=str(task.result))
        elif task.state == 'SUCCESS':
            result = task.get()
            # Check if the conversion result contains an error
            if result.get('error'):
                return ConversionJobResult(job_id=job_id, status="FAILURE", error=result['error'])
            
            # Ensure page_content is properly handled as a dictionary
            if 'page_content' in result and result['page_content'] is not None:
                if not isinstance(result['page_content'], dict):
                    # If page_content is not a dict, set it to None to avoid type errors
                    result['page_content'] = None
                    logging.warning(f"Invalid page_content type in job {job_id}, expected dict but got {type(result['page_content'])}")
            
            conversion_result = ConversionResult(**result)
            
            # If page numbers are requested, format the markdown with page numbers
            if include_page_numbers and conversion_result.page_content and not conversion_result.error:
                conversion_result.markdown = self.get_markdown_with_page_numbers(conversion_result)
                
            return ConversionJobResult(job_id=job_id, status="SUCCESS", result=conversion_result)
        else:
            return ConversionJobResult(job_id=job_id, status="FAILURE", error=str(task.result))

    def get_batch_conversion_task_result(self, job_id: str, include_page_numbers: bool = False) -> BatchConversionJobResult:
        """Get the status and results of a batch document conversion job.

        Args:
            job_id: The ID of the batch job
            include_page_numbers: Whether to include page numbers in the markdown

        Returns:
            BatchConversionJobResult: The result of the batch conversion job
        """
        # Import celery_app only when needed to avoid circular imports
        from worker.celery_config import celery_app

        task = AsyncResult(job_id, app=celery_app)
        if task.state == 'PENDING':
            return BatchConversionJobResult(job_id=job_id, status="IN_PROGRESS")

        elif task.state == 'SUCCESS':
            batch_results = task.get()
            conversion_results = []

            for result in batch_results:
                if result.get('error'):
                    conversion_results.append(
                        ConversionJobResult(
                            job_id=job_id,
                            status="FAILURE",
                            error=result['error']
                        )
                    )
                else:
                    # Ensure page_content is properly handled as a dictionary
                    if 'page_content' in result and result['page_content'] is not None:
                        if not isinstance(result['page_content'], dict):
                            # If page_content is not a dict, set it to None to avoid type errors
                            result['page_content'] = None
                            logging.warning(f"Invalid page_content type in batch job {job_id}, expected dict but got {type(result['page_content'])}")
                    
                    conversion_result = ConversionResult(**result)
                    
                    # If page numbers are requested, format the markdown with page numbers
                    if include_page_numbers and conversion_result.page_content and not conversion_result.error:
                        conversion_result.markdown = self.get_markdown_with_page_numbers(conversion_result)
                        
                    conversion_results.append(
                        ConversionJobResult(
                            job_id=job_id,
                            status="SUCCESS",
                            result=conversion_result
                        )
                    )

            return BatchConversionJobResult(
                job_id=job_id,
                status="SUCCESS",
                conversion_results=conversion_results
            )
        else:
            return BatchConversionJobResult(
                job_id=job_id,
                status="FAILURE",
                error=str(task.result))

    def chunk_document_from_job(
        self, 
        job_id: str, 
        max_tokens: int = 512, 
        merge_peers: bool = True,
        include_page_numbers: bool = False
    ) -> ChunkingResult:
        """
        Chunk a document from a conversion job.

        Args:
            job_id: The ID of the conversion job
            max_tokens: Maximum number of tokens per chunk
            merge_peers: Whether to merge undersized peer chunks
            include_page_numbers: Whether to include page number references in chunk metadata

        Returns:
            ChunkingResult: The chunking result
        """
        # Get the conversion result first
        job_result = self.get_single_document_task_result(job_id, include_page_numbers=include_page_numbers)
        
        if job_result.status != "SUCCESS" or not job_result.result:
            return ChunkingResult(
                job_id=job_id,
                filename="unknown",
                error=f"Job failed or not completed: {job_result.error if job_result.error else 'Unknown error'}"
            )
            
        # Initialize the chunker
        chunker = SDPMChunker(
            embedding_model="minishlab/potion-base-8M",
            threshold=0.5,                              # Similarity threshold (0-1)
            chunk_size=512,                             # Maximum tokens per chunk
            min_sentences=1,                            # Initial sentences per chunk
            skip_window=1                               # Number of chunks to skip when looking for similaritie
        )
        
        try:
            # Get the text content
            text = job_result.result.markdown
            filename = job_result.result.filename
            
            # Process the text through the chunker
            chunk_results = chunker(text)
            
            # Convert chunker results to our Chunk model
            chunks = []
            current_page = None
            
            if include_page_numbers and job_result.result.page_content:
                # Create a mapping of text positions to page numbers
                page_map = {}
                current_pos = 0
                
                for page_num, content in sorted(job_result.result.page_content.items()):
                    content_len = len(content)
                    page_map[(current_pos, current_pos + content_len)] = int(page_num)
                    current_pos += content_len
                    
            for chunk_result in chunk_results:
                chunk_metadata = {
                    "token_count": str(chunk_result.token_count),
                    "sentence_count": str(len(chunk_result.sentences))
                }
                
                # If page numbers are requested and we have page content
                if include_page_numbers and job_result.result.page_content:
                    # Find the page numbers for this chunk
                    chunk_pages = set()
                    chunk_start = text.find(chunk_result.text)
                    chunk_end = chunk_start + len(chunk_result.text)
                    
                    for (start, end), page in page_map.items():
                        if (chunk_start < end and chunk_end > start):
                            chunk_pages.add(page)
                    
                    if chunk_pages:
                        chunk_metadata["start_page"] = str(min(chunk_pages))
                        chunk_metadata["end_page"] = str(max(chunk_pages))
                
                chunks.append(Chunk(
                    text=chunk_result.text,
                    metadata=chunk_metadata,
                    page_numbers=sorted(list(chunk_pages)) if include_page_numbers and chunk_pages else None,
                    start_page=int(chunk_metadata["start_page"]) if "start_page" in chunk_metadata else None,
                    end_page=int(chunk_metadata["end_page"]) if "end_page" in chunk_metadata else None
                ))
            
            return ChunkingResult(
                job_id=job_id,
                filename=filename,
                chunks=chunks
            )
            
        except Exception as e:
            logging.error(f"Error chunking document from job {job_id}: {str(e)}")
            return ChunkingResult(
                job_id=job_id,
                filename=filename if 'filename' in locals() else "unknown",
                error=f"Error during chunking: {str(e)}"
            )

    def chunk_text_directly(
        self, 
        text: str, 
        filename: str = "input.txt", 
        max_tokens: int = 512, 
        merge_peers: bool = True,
        include_page_numbers: bool = False
    ) -> ChunkingResult:
        """
        Chunk text directly without requiring a conversion job.
        
        Args:
            text: The text content to chunk
            filename: A name to identify the source
            max_tokens: Maximum number of tokens per chunk
            merge_peers: Whether to merge undersized peer chunks
            include_page_numbers: Whether to include page number references in chunk metadata
            
        Returns:
            ChunkingResult: The chunking result
        """
        # Initialize the chunker
        chunker = SDPMChunker(
            embedding_model="minishlab/potion-base-8M",
            threshold=0.5,                              # Similarity threshold (0-1)
            chunk_size=512,                             # Maximum tokens per chunk
            min_sentences=1,                            # Initial sentences per chunk
            skip_window=1                               # Number of chunks to skip when looking for similaritie
        )
        
        try:
            # Process the text through the chunker
            chunk_results = chunker(text)
            
            # Convert chunker results to our Chunk model
            chunks = []
            for chunk_result in chunk_results:
                chunk_metadata = {
                    "token_count": str(chunk_result.token_count),
                    "sentence_count": str(len(chunk_result.sentences))
                }
                
                chunks.append(Chunk(
                    text=chunk_result.text,
                    metadata=chunk_metadata,
                    page_numbers=None,  # No page numbers for direct text chunking
                    start_page=None,
                    end_page=None
                ))
            
            return ChunkingResult(
                job_id=str(uuid.uuid4()),  # Generate a new ID for direct chunking
                filename=filename,
                chunks=chunks
            )
            
        except Exception as e:
            logging.error(f"Error chunking text directly: {str(e)}")
            return ChunkingResult(
                job_id=str(uuid.uuid4()),
                filename=filename,
                error=f"Error during chunking: {str(e)}"
            )

    def convert_document_with_pages(self, document: Tuple[str, BytesIO], **kwargs) -> ConversionResult:
        """
        Convert a document and include page-by-page content in the result.
        
        Args:
            document: A tuple containing the filename and file content
            **kwargs: Additional arguments to pass to the document converter
            
        Returns:
            ConversionResult: The conversion result with page-by-page content
        """
        result = self.document_converter.convert(document, **kwargs)
        if result.error:
            logging.error(f"Failed to convert {document[0]}: {result.error}")
            raise HTTPException(status_code=500, detail=result.error)
        return result

    def get_markdown_with_page_numbers(self, result: ConversionResult) -> str:
        """
        Format the conversion result as markdown with page numbers.
        
        Args:
            result: The conversion result
            
        Returns:
            str: Markdown content with page numbers
        """
        if not result.page_content:
            return result.markdown or ""
            
        formatted_content = []
        for page_num, content in sorted(result.page_content.items()):
            if content.strip():
                formatted_content.append(f"## Page {page_num}\n\n{content}\n")
                
        return "\n".join(formatted_content)
