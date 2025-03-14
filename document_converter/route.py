from io import BytesIO
from typing import List, Optional
from fastapi import APIRouter, File, HTTPException, UploadFile, Query, status
from fastapi.responses import JSONResponse
import logging

from document_converter.schema import (
    BatchConversionJobResult,
    ConversionJobResult,
    ConversionResult,
    ChunkingResult,
    TextChunkingRequest,
    HealthCheckResponse
)
from document_converter.service import DocumentConverterService, DoclingDocumentConversion
from document_converter.utils import is_file_format_supported
from worker.tasks import convert_document_task, convert_documents_task, ping

router = APIRouter()

# Could be docling or another converter as long as it implements DocumentConversionBase
converter = DoclingDocumentConversion()
document_converter_service = DocumentConverterService(document_converter=converter)


# Document direct conversion endpoints
@router.post(
    '/documents/convert',
    response_model=ConversionResult,
    response_model_exclude_unset=True,
    status_code=status.HTTP_200_OK,
    responses={
        200: {"description": "Document successfully converted"},
        400: {"description": "Invalid request or unsupported file format"},
        500: {"description": "Internal server error during conversion"}
    },
    description="Convert a single document synchronously",
)
async def convert_single_document(
    document: UploadFile = File(..., description="The document file to convert"),
    extract_tables_as_images: bool = Query(
        False,
        description="Whether to extract tables as images"
    ),
    image_resolution_scale: int = Query(
        4,
        ge=1,
        le=4,
        description="Scale factor for image resolution (1-4)"
    ),
    include_page_numbers: bool = Query(
        False,
        description="Whether to include page numbers in the markdown"
    ),
):
    try:
        # Read the file content
        file_content = await document.read()
        
        # Convert the document
        result = document_converter_service.convert_document(
            document=(document.filename, BytesIO(file_content)),
            extract_tables=extract_tables_as_images,
            image_resolution_scale=image_resolution_scale,
            include_page_numbers=include_page_numbers,
        )
        
        # Return the result
        return result
    except Exception as e:
        logging.error(f"Error in convert_single_document: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error converting document: {str(e)}"
        )


@router.post(
    '/documents/batch-convert',
    response_model=List[ConversionResult],
    response_model_exclude_unset=True,
    status_code=status.HTTP_200_OK,
    responses={
        200: {"description": "All documents successfully converted"},
        400: {"description": "Invalid request or unsupported file format"},
        500: {"description": "Internal server error during conversion"}
    },
    description="Convert multiple documents synchronously",
)
async def convert_multiple_documents(
    documents: List[UploadFile] = File(..., description="List of document files to convert"),
    extract_tables_as_images: bool = Query(
        False,
        description="Whether to extract tables as images"
    ),
    image_resolution_scale: int = Query(
        4,
        ge=1,
        le=4,
        description="Scale factor for image resolution (1-4)"
    ),
    include_page_numbers: bool = Query(
        True,
        description="Whether to include page numbers in the markdown"
    ),
):
    try:
        # Read all files and prepare for batch conversion
        document_data = []
        for document in documents:
            file_content = await document.read()
            document_data.append((document.filename, BytesIO(file_content)))
        
        # Convert all documents
        results = document_converter_service.convert_documents(
            documents=document_data,
            extract_tables=extract_tables_as_images,
            image_resolution_scale=image_resolution_scale,
            include_page_numbers=include_page_numbers,
        )
        
        # Return the results
        return results
    except Exception as e:
        logging.error(f"Error in convert_multiple_documents: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error converting documents: {str(e)}"
        )


# Asynchronous conversion jobs endpoints
@router.post(
    '/conversion-jobs',
    response_model=ConversionJobResult,
    status_code=status.HTTP_202_ACCEPTED,
    responses={
        202: {"description": "Conversion job accepted and queued"},
        400: {"description": "Invalid request or unsupported file format"},
        500: {"description": "Failed to queue conversion job"}
    },
    description="Create an asynchronous conversion job for a single document",
)
async def create_single_document_conversion_job(
    document: UploadFile = File(..., description="The document file to convert"),
    extract_tables_as_images: bool = Query(
        False,
        description="Whether to extract tables as images"
    ),
    image_resolution_scale: int = Query(
        4,
        ge=1,
        le=4,
        description="Scale factor for image resolution (1-4)"
    ),
    include_page_numbers: bool = Query(
        True,
        description="Whether to include page numbers in the markdown"
    ),
):
    try:
        # Read the file content
        file_content = await document.read()
        
        # Import the task function
        from worker.tasks import convert_document_task
        
        # Queue the conversion task
        task = convert_document_task.delay(
            document=(document.filename, file_content),
            extract_tables=extract_tables_as_images,
            image_resolution_scale=image_resolution_scale,
            include_page_numbers=include_page_numbers,
        )

        return ConversionJobResult(
            job_id=task.id,
            status="IN_PROGRESS"
        )
    except Exception as e:
        logging.error(f"Error in create_single_document_conversion_job: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error creating conversion job: {str(e)}"
        )


@router.get(
    '/conversion-jobs/{job_id}',
    response_model=ConversionJobResult,
    responses={
        200: {"description": "Conversion job completed successfully"},
        202: {"description": "Conversion job is still in progress"},
        404: {"description": "Job not found"},
        422: {"description": "Conversion job failed"}
    },
    description="Get the status and result of a single document conversion job",
)
async def get_conversion_job_status(
    job_id: str,
    include_page_numbers: bool = Query(
        True,
        description="Whether to include page numbers in the markdown"
    ),
):
    try:
        # Attempt to get the job status and result
        result = document_converter_service.get_single_document_task_result(
            job_id=job_id,
            include_page_numbers=include_page_numbers,
        )
        
        # Return 202 Accepted if job is still in progress
        if result.status in ["IN_PROGRESS"]:
            return JSONResponse(
                status_code=status.HTTP_202_ACCEPTED,
                content=result.model_dump(exclude_none=True)
            )
            
        # Return 422 for failed jobs
        if result.status == "FAILURE":
            return JSONResponse(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                content=result.model_dump(exclude_none=True)
            )
            
        # Return 200 OK for successful jobs
        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content=result.model_dump(exclude_none=True)
        )
    except KeyError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job not found: {job_id}"
        )


@router.post(
    '/batch-conversion-jobs',
    response_model=BatchConversionJobResult,
    status_code=status.HTTP_202_ACCEPTED,
    responses={
        202: {"description": "Batch conversion job accepted and queued"},
        400: {"description": "Invalid request or unsupported file format"},
        500: {"description": "Failed to queue batch conversion job"}
    },
    description="Create an asynchronous conversion job for multiple documents",
)
async def create_batch_conversion_job(
    documents: List[UploadFile] = File(..., description="List of document files to convert"),
    extract_tables_as_images: bool = Query(
        False,
        description="Whether to extract tables as images"
    ),
    image_resolution_scale: int = Query(
        4,
        ge=1,
        le=4,
        description="Scale factor for image resolution (1-4)"
    ),
    include_page_numbers: bool = Query(
        True,
        description="Whether to include page numbers in the markdown"
    ),
):
    try:
        # Read all files and prepare for batch conversion
        document_data = []
        for document in documents:
            file_content = await document.read()
            document_data.append((document.filename, file_content))
        
        # Import the task function
        from worker.tasks import convert_documents_task
        
        # Queue the batch conversion task
        task = convert_documents_task.delay(
            documents=document_data,
            extract_tables=extract_tables_as_images,
            image_resolution_scale=image_resolution_scale,
            include_page_numbers=include_page_numbers,
        )

        return BatchConversionJobResult(
            job_id=task.id,
            status="IN_PROGRESS"
        )
    except Exception as e:
        logging.error(f"Error in create_batch_conversion_job: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error creating batch conversion job: {str(e)}"
        )


@router.get(
    '/batch-conversion-jobs/{job_id}',
    response_model=BatchConversionJobResult,
    responses={
        200: {"description": "All conversion jobs completed successfully"},
        202: {"description": "Batch job is still in progress"},
        404: {"description": "Batch job not found"},
        422: {"description": "Batch job failed"}
    },
    description="Get the status and results of a batch conversion job",
)
async def get_batch_conversion_job_status(
    job_id: str,
    include_page_numbers: bool = Query(
        True,
        description="Whether to include page numbers in the markdown"
    ),
):
    try:
        # Attempt to get the batch job status and results
        result = document_converter_service.get_batch_conversion_task_result(
            job_id=job_id,
            include_page_numbers=include_page_numbers,
        )
        
        # Return 202 Accepted if the batch job or any sub-job is still in progress
        if result.status in ["IN_PROGRESS"] or any(
            job.status in ["IN_PROGRESS"]
            for job in result.conversion_results
        ):
            return JSONResponse(
                status_code=status.HTTP_202_ACCEPTED,
                content=result.model_dump(exclude_none=True)
            )
            
        # Return 422 for failed batch jobs
        if result.status == "FAILURE" or any(
            job.status == "FAILURE"
            for job in result.conversion_results
        ):
            return JSONResponse(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                content=result.model_dump(exclude_none=True)
            )
            
        # Return 200 OK for successful batch jobs (all success)
        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content=result.model_dump(exclude_none=True)
        )
    except KeyError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Batch job not found: {job_id}"
        )


@router.get(
    "/health",
    response_model=HealthCheckResponse,
    responses={
        200: {"description": "All services are healthy"},
        500: {"description": "One or more services are unhealthy"}
    },
    description="Check the health status of all dependent services"
)
async def health_check():
    try:
        # Check Celery/Redis connection by sending a ping task
        result = ping.delay()
        response = result.get(timeout=3)  # Wait up to 3 seconds for response
        
        if response != "pong":
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Celery/Redis connection test failed"
            )
        
        return {
            "status": "healthy",
            "services": {
                "celery": "connected",
                "redis": "connected",
                "docling": "connected",
                "document_converter": "connected",
            }
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.get(
    '/conversion-jobs/{job_id}/chunks',
    response_model=ChunkingResult,
    responses={
        200: {"description": "Document chunked successfully"},
        500: {"description": "Internal server error"}
    },
    description="Chunk a converted document using a completed job ID with Semantic Double-Pass Merging",
)
async def chunk_document_from_job(
    job_id: str,
    max_tokens: int = Query(
        512,
        ge=64,
        le=2048,
        description="Maximum number of tokens per chunk (used as chunk_size in SDPMChunker)"
    ),
    merge_peers: bool = Query(
        True,
        description="Whether to merge undersized peer chunks (used for internal configuration)"
    ),
    include_page_numbers: bool = Query(
        True,
        description="Whether to include page number references in chunk metadata"
    ),
):
    try:
        # Attempt to get the chunking result
        result = document_converter_service.chunk_document_from_job(
            job_id=job_id,
            max_tokens=max_tokens,
            merge_peers=merge_peers,
            include_page_numbers=include_page_numbers,
        )
        
        # Return the chunking result
        if result.error:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=result.error
            )

        return result
    except Exception as e:
        logging.error(f"Error in chunk_document_from_job: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error chunking document: {str(e)}"
        )


@router.get(
    '/batch-conversion-jobs/{job_id}/chunks',
    response_model=List[ChunkingResult],
    responses={
        200: {"description": "Documents chunked successfully"},
        404: {"description": "Batch job not found"},
        500: {"description": "Internal server error"}
    },
    description="Chunk all converted documents from a completed batch job using Semantic Double-Pass Merging",
)
async def chunk_batch_documents_from_job(
    job_id: str,
    max_tokens: int = Query(
        512,
        ge=64,
        le=2048,
        description="Maximum number of tokens per chunk (used as chunk_size in SDPMChunker)"
    ),
    merge_peers: bool = Query(
        True,
        description="Whether to merge undersized peer chunks (used for internal configuration)"
    ),
    include_page_numbers: bool = Query(
        True,
        description="Whether to include page number references in chunk metadata"
    ),
):
    try:
        # Attempt to chunk all documents from the batch job
        results = document_converter_service.chunk_batch_documents_from_job(
            job_id=job_id,
            max_tokens=max_tokens,
            merge_peers=merge_peers,
            include_page_numbers=include_page_numbers,
        )
        
        # Return the chunking results
        return results
    except KeyError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Batch job not found: {job_id}"
        )
    except Exception as e:
        logging.error(f"Error in chunk_batch_documents_from_job: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error chunking documents: {str(e)}"
        )


@router.post(
    '/text/chunk',
    response_model=ChunkingResult,
    responses={
        200: {"description": "Text successfully chunked"},
        422: {"description": "Error during chunking"}
    },
    description="Chunk text directly without requiring a conversion job",
)
async def chunk_text_directly(
    request: TextChunkingRequest,
):
    try:
        # Attempt to chunk the text directly
        result = document_converter_service.chunk_text_directly(
            text=request.text,
            filename=request.filename,
            max_tokens=request.max_tokens,
            merge_peers=request.merge_peers,
            include_page_numbers=request.include_page_numbers,
        )
        
        # Return the chunking result
        if result.error:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=result.error
            )

        return result
    except Exception as e:
        logging.error(f"Error in chunk_text_directly: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error chunking text: {str(e)}"
        )
