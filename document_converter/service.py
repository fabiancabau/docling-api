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
from transformers import AutoTokenizer
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
    def _process_document_images(conv_res) -> Tuple[str, List[ImageData]]:
        images = []
        table_counter = 0
        picture_counter = 0
        content_md = conv_res.document.export_to_markdown(image_mode=ImageRefMode.PLACEHOLDER)

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

                image_bytes = base64.b64encode(img_buffer.getvalue()).decode('utf-8')
                images.append(ImageData(type=image_type, filename=image_name, image=image_bytes))

        return content_md, images

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

        content_md, images = self._process_document_images(conv_res)
        return ConversionResult(filename=doc_filename, markdown=content_md, images=images)

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

            content_md, images = self._process_document_images(conv_res)
            results.append(ConversionResult(filename=doc_filename, markdown=content_md, images=images))

        return results


class DocumentConverterService:
    def __init__(self, document_converter: DocumentConversionBase):
        self.document_converter = document_converter

    def convert_document(self, document: Tuple[str, BytesIO], **kwargs) -> ConversionResult:
        result = self.document_converter.convert(document, **kwargs)
        if result.error:
            logging.error(f"Failed to convert {document[0]}: {result.error}")
            raise HTTPException(status_code=500, detail=result.error)
        return result

    def convert_documents(self, documents: List[Tuple[str, BytesIO]], **kwargs) -> List[ConversionResult]:
        return self.document_converter.convert_batch(documents, **kwargs)

    def convert_document_task(
        self,
        document: Tuple[str, bytes],
        **kwargs,
    ) -> ConversionResult:
        document = (document[0], BytesIO(document[1]))
        return self.document_converter.convert(document, **kwargs)

    def convert_documents_task(
        self,
        documents: List[Tuple[str, bytes]],
        **kwargs,
    ) -> List[ConversionResult]:
        documents = [(filename, BytesIO(file)) for filename, file in documents]
        return self.document_converter.convert_batch(documents, **kwargs)

    def get_single_document_task_result(self, job_id: str) -> ConversionJobResult:
        """Get the status and result of a document conversion job.

        Returns:
        - IN_PROGRESS: When task is still running
        - SUCCESS: When conversion completed successfully
        - FAILURE: When task failed or conversion had errors
        """
        # Import celery_app only when needed to avoid circular imports
        from worker.celery_config import celery_app

        task = AsyncResult(job_id, app=celery_app)
        if task.state == 'PENDING':
            return ConversionJobResult(job_id=job_id, status="IN_PROGRESS")

        elif task.state == 'SUCCESS':
            result = task.get()
            # Check if the conversion result contains an error
            if result.get('error'):
                return ConversionJobResult(job_id=job_id, status="FAILURE", error=result['error'])

            return ConversionJobResult(job_id=job_id, status="SUCCESS", result=ConversionResult(**result))

        else:
            return ConversionJobResult(job_id=job_id, status="FAILURE", error=str(task.result))

    def get_batch_conversion_task_result(self, job_id: str) -> BatchConversionJobResult:
        """Get the status and results of a batch conversion job.

        Returns:
        - IN_PROGRESS: When task is still running
        - SUCCESS: A batch is successful as long as the task is successful
        - FAILURE: When the task fails for any reason
        """
        # Import celery_app only when needed to avoid circular imports
        from worker.celery_config import celery_app

        task = AsyncResult(job_id, app=celery_app)
        if task.state == 'PENDING':
            return BatchConversionJobResult(job_id=job_id, status="IN_PROGRESS")

        # Task completed successfully, but need to check individual conversion results
        if task.state == 'SUCCESS':
            conversion_results = task.get()
            job_results = []

            for result in conversion_results:
                if result.get('error'):
                    job_result = ConversionJobResult(status="FAILURE", error=result['error'])
                else:
                    job_result = ConversionJobResult(
                        status="SUCCESS", result=ConversionResult(**result).model_dump(exclude_unset=True)
                    )
                job_results.append(job_result)

            return BatchConversionJobResult(job_id=job_id, status="SUCCESS", conversion_results=job_results)

        return BatchConversionJobResult(job_id=job_id, status="FAILURE", error=str(task.result))

    def chunk_document_from_job(self, job_id: str, max_tokens: int = 512, merge_peers: bool = True) -> ChunkingResult:
        """
        Retrieve a completed conversion job and chunk the resulting document.
        
        Args:
            job_id: The ID of the completed conversion job
            max_tokens: Maximum number of tokens per chunk
            merge_peers: Whether to merge undersized peer chunks
            
        Returns:
            ChunkingResult containing the chunks extracted from the document
        """
        # Get the conversion job result
        job_result = self.get_single_document_task_result(job_id)
        
        # Check if the job is completed successfully
        if job_result.status != "SUCCESS":
            return ChunkingResult(
                job_id=job_id,
                filename=job_result.result.filename if job_result.result else "unknown",
                error=f"Cannot chunk document: job status is {job_result.status}. {job_result.error or ''}"
            )
        
        # Access the markdown content
        result = job_result.result
        if not result or not result.markdown:
            return ChunkingResult(
                job_id=job_id,
                filename=result.filename if result else "unknown",
                error="Cannot chunk document: no markdown content available"
            )
            
        try:
            # Convert markdown to a DoclingDocument
            from docling.document_converter import DocumentConverter
            from docling.datamodel.base_models import DocumentStream
            
            # Create a document stream from the markdown
            markdown_stream = DocumentStream(
                name=f"{result.filename}.md",
                stream=io.BytesIO(result.markdown.encode('utf-8'))
            )
            
            # Convert the markdown to a DoclingDocument
            converter = DocumentConverter()
            doc_result = converter.convert(markdown_stream)
            
            if doc_result.errors:
                return ChunkingResult(
                    job_id=job_id,
                    filename=result.filename,
                    error=f"Error creating DoclingDocument: {doc_result.errors[0].error_message}"
                )
                
            docling_doc = doc_result.document
            
            # Initialize the SDPMChunker with the specified parameters
            # We're using the default embedding model "minishlab/potion-base-8M"
            chunker = SDPMChunker(
                chunk_size=max_tokens,
                threshold=0.5,  # Similarity threshold (0-1)
                min_sentences=1,  # Initial sentences per chunk
                skip_window=1     # Number of chunks to skip when looking for similarities
            )
            
            # Perform chunking
            chunks = []
            try:
                # Extract text content from docling_doc
                # The DoclingDocument doesn't have a 'text' attribute directly
                # Instead, we'll extract it from the markdown content
                text_content = result.markdown  # Use the markdown content directly
                
                # Chunk the text using SDPMChunker
                chonkie_chunks = chunker.chunk(text_content)
                
                for chunk in chonkie_chunks:
                    # Get the plain text from the chunk
                    plain_text = chunk.text
                    
                    # Create additional metadata dictionary
                    additional_metadata = {
                        "token_count": chunk.token_count,
                        "start_index": chunk.start_index,
                        "end_index": chunk.end_index
                    }
                    
                    # Add sentence information if available
                    if hasattr(chunk, "sentences") and chunk.sentences:
                        additional_metadata["sentence_count"] = len(chunk.sentences)
                    
                    chunks.append(Chunk(
                        text=plain_text,
                        metadata=additional_metadata
                    ))
            except Exception as e:
                logging.error(f"Error during chunking process: {str(e)}")
                return ChunkingResult(
                    job_id=job_id,
                    filename=result.filename,
                    error=f"Error during chunking process: {str(e)}"
                )
                
            return ChunkingResult(
                job_id=job_id,
                filename=result.filename,
                chunks=chunks
            )
            
        except Exception as e:
            logging.error(f"Error chunking document: {str(e)}")
            return ChunkingResult(
                job_id=job_id,
                filename=result.filename,
                error=f"Error chunking document: {str(e)}"
            )

    def chunk_batch_documents_from_job(self, job_id: str, max_tokens: int = 512, merge_peers: bool = True) -> List[ChunkingResult]:
        """
        Retrieve a completed batch conversion job and chunk all the resulting documents.
        
        Args:
            job_id: The ID of the completed batch conversion job
            max_tokens: Maximum number of tokens per chunk
            merge_peers: Whether to merge undersized peer chunks
            
        Returns:
            List of ChunkingResult containing the chunks extracted from each document
        """
        # Get the batch conversion job result
        batch_result = self.get_batch_conversion_task_result(job_id)
        
        # Check if the batch job is completed successfully
        if batch_result.status != "SUCCESS":
            return [ChunkingResult(
                job_id=job_id,
                filename="batch",
                error=f"Cannot chunk documents: batch job status is {batch_result.status}. {batch_result.error or ''}"
            )]
        
        chunking_results = []
        
        # Process each document in the batch
        for job_result in batch_result.conversion_results:
            if job_result.status != "SUCCESS" or not job_result.result:
                # Skip failed jobs
                chunking_results.append(ChunkingResult(
                    job_id=job_id,
                    filename=job_result.result.filename if job_result.result else "unknown",
                    error=f"Cannot chunk document: job status is {job_result.status}. {job_result.error or ''}"
                ))
                continue
                
            result = job_result.result
            if not result.markdown:
                chunking_results.append(ChunkingResult(
                    job_id=job_id,
                    filename=result.filename,
                    error="Cannot chunk document: no markdown content available"
                ))
                continue
                
            try:
                # Convert markdown to a DoclingDocument
                from docling.document_converter import DocumentConverter
                from docling.datamodel.base_models import DocumentStream
                
                # Create a document stream from the markdown
                markdown_stream = DocumentStream(
                    name=f"{result.filename}.md",
                    stream=io.BytesIO(result.markdown.encode('utf-8'))
                )
                
                # Convert the markdown to a DoclingDocument
                converter = DocumentConverter()
                doc_result = converter.convert(markdown_stream)
                
                if doc_result.errors:
                    chunking_results.append(ChunkingResult(
                        job_id=job_id,
                        filename=result.filename,
                        error=f"Error creating DoclingDocument: {doc_result.errors[0].error_message}"
                    ))
                    continue
                    
                docling_doc = doc_result.document
                
                # Initialize the SDPMChunker with the specified parameters
                # We're using the default embedding model "minishlab/potion-base-8M"
                chunker = SDPMChunker(
                    chunk_size=max_tokens,
                    threshold=0.5,  # Similarity threshold (0-1)
                    min_sentences=1,  # Initial sentences per chunk
                    skip_window=1     # Number of chunks to skip when looking for similarities
                )
                
                # Perform chunking
                chunks = []
                try:
                    # Extract text content from docling_doc
                    # The DoclingDocument doesn't have a 'text' attribute directly
                    # Instead, we'll extract it from the markdown content
                    text_content = result.markdown  # Use the markdown content directly
                    
                    # Chunk the text using SDPMChunker
                    chonkie_chunks = chunker.chunk(text_content)
                    
                    for chunk in chonkie_chunks:
                        # Get the plain text from the chunk
                        plain_text = chunk.text
                        
                        # Create additional metadata dictionary
                        additional_metadata = {
                            "token_count": chunk.token_count,
                            "start_index": chunk.start_index,
                            "end_index": chunk.end_index
                        }
                        
                        # Add sentence information if available
                        if hasattr(chunk, "sentences") and chunk.sentences:
                            additional_metadata["sentence_count"] = len(chunk.sentences)
                        
                        chunks.append(Chunk(
                            text=plain_text,
                            metadata=additional_metadata
                        ))
                except Exception as e:
                    logging.error(f"Error during chunking process: {str(e)}")
                    chunking_results.append(ChunkingResult(
                        job_id=job_id,
                        filename=result.filename,
                        error=f"Error during chunking process: {str(e)}"
                    ))
                    continue
                
                chunking_results.append(ChunkingResult(
                    job_id=job_id,
                    filename=result.filename,
                    chunks=chunks
                ))
                
            except Exception as e:
                logging.error(f"Error chunking document {result.filename}: {str(e)}")
                chunking_results.append(ChunkingResult(
                    job_id=job_id,
                    filename=result.filename,
                    error=f"Error chunking document: {str(e)}"
                ))
                
        return chunking_results

    def chunk_text_directly(self, text: str, filename: str = "input.txt", max_tokens: int = 512, merge_peers: bool = True) -> ChunkingResult:
        """
        Chunk text directly without requiring a conversion job.
        
        Args:
            text: The text content to chunk
            filename: A name to identify the source (for reporting purposes)
            max_tokens: Maximum number of tokens per chunk
            merge_peers: Whether to merge undersized peer chunks
            
        Returns:
            ChunkingResult containing the chunks extracted from the text
        """
        try:
            # Initialize the SDPMChunker with the specified parameters
            chunker = SDPMChunker(
                chunk_size=max_tokens,
                threshold=0.5,  # Similarity threshold (0-1)
                min_sentences=1,  # Initial sentences per chunk
                skip_window=1     # Number of chunks to skip when looking for similarities
            )
            
            # Perform chunking
            chunks = []
            try:
                # Chunk the text using SDPMChunker
                chonkie_chunks = chunker.chunk(text)
                
                for chunk in chonkie_chunks:
                    # Get the plain text from the chunk
                    plain_text = chunk.text
                    
                    # Create additional metadata dictionary
                    additional_metadata = {
                        "token_count": chunk.token_count,
                        "start_index": chunk.start_index,
                        "end_index": chunk.end_index
                    }
                    
                    # Add sentence information if available
                    if hasattr(chunk, "sentences") and chunk.sentences:
                        additional_metadata["sentence_count"] = len(chunk.sentences)
                    
                    chunks.append(Chunk(
                        text=plain_text,
                        metadata=additional_metadata
                    ))
            except Exception as e:
                logging.error(f"Error during chunking process: {str(e)}")
                return ChunkingResult(
                    job_id="direct",
                    filename=filename,
                    error=f"Error during chunking process: {str(e)}"
                )
                
            return ChunkingResult(
                job_id="direct",
                filename=filename,
                chunks=chunks
            )
            
        except Exception as e:
            logging.error(f"Error chunking text: {str(e)}")
            return ChunkingResult(
                job_id="direct",
                filename=filename,
                error=f"Error chunking text: {str(e)}"
            )
