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
    TextChunkingRequest
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
):
    file_bytes = await document.read()
    if not is_file_format_supported(file_bytes, document.filename):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file format: {document.filename}"
        )

    return document_converter_service.convert_document(
        (document.filename, BytesIO(file_bytes)),
        extract_tables=extract_tables_as_images,
        image_resolution_scale=image_resolution_scale,
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
):
    doc_streams = []
    for document in documents:
        file_bytes = await document.read()
        if not is_file_format_supported(file_bytes, document.filename):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Unsupported file format: {document.filename}"
            )
        doc_streams.append((document.filename, BytesIO(file_bytes)))

    return document_converter_service.convert_documents(
        doc_streams,
        extract_tables=extract_tables_as_images,
        image_resolution_scale=image_resolution_scale,
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
):
    file_bytes = await document.read()
    if not is_file_format_supported(file_bytes, document.filename):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file format: {document.filename}"
        )

    task = convert_document_task.delay(
        (document.filename, file_bytes),
        extract_tables=extract_tables_as_images,
        image_resolution_scale=image_resolution_scale,
    )

    return ConversionJobResult(
        job_id=task.id,
        status="IN_PROGRESS"
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
async def get_conversion_job_status(job_id: str):
    try:
        result = document_converter_service.get_single_document_task_result(job_id)
        
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
):
    doc_data = []
    for document in documents:
        file_bytes = await document.read()
        if not is_file_format_supported(file_bytes, document.filename):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Unsupported file format: {document.filename}"
            )
        doc_data.append((document.filename, file_bytes))

    task = convert_documents_task.delay(
        doc_data,
        extract_tables=extract_tables_as_images,
        image_resolution_scale=image_resolution_scale,
    )

    return BatchConversionJobResult(
        job_id=task.id,
        status="IN_PROGRESS"
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
async def get_batch_conversion_job_status(job_id: str):
    try:
        result = document_converter_service.get_batch_conversion_task_result(job_id)
        
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
        200: {"description": "Document successfully chunked"},
        400: {"description": "Invalid request or chunking parameters"},
        404: {"description": "Job not found"},
        422: {"description": "Job failed or is not completed yet"}
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
        False,
        description="Whether to include page number references in chunk metadata (when available)"
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
        
        # Return error response if there's an error
        if result.error:
            return JSONResponse(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                content=result.model_dump(exclude_none=True)
            )
            
        # Return 200 OK for successful chunking
        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content=result.model_dump(exclude_none=True)
        )
    except KeyError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job not found: {job_id}"
        )
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
        200: {"description": "Documents successfully chunked"},
        400: {"description": "Invalid request or chunking parameters"},
        404: {"description": "Batch job not found"},
        422: {"description": "Batch job failed or is not completed yet"}
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
        False,
        description="Whether to include page number references in chunk metadata (when available)"
    ),
):
    try:
        # Attempt to get the chunking results
        results = document_converter_service.chunk_batch_documents_from_job(
            job_id=job_id,
            max_tokens=max_tokens,
            merge_peers=merge_peers,
            include_page_numbers=include_page_numbers,
        )
        
        # Check if there were errors in the batch
        all_errors = all(result.error is not None for result in results)
        
        if all_errors:
            # All documents had errors
            return JSONResponse(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                content=[result.model_dump(exclude_none=True) for result in results]
            )
            
        # Return 200 OK for at least partial success
        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content=[result.model_dump(exclude_none=True) for result in results]
        )
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
        400: {"description": "Invalid request or chunking parameters"},
        422: {"description": "Error during text chunking"}
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
        
        # Return error response if there's an error
        if result.error:
            return JSONResponse(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                content=result.model_dump(exclude_none=True)
            )
            
        # Return 200 OK for successful chunking
        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content=result.model_dump(exclude_none=True)
        )
    except Exception as e:
        logging.error(f"Error in chunk_text_directly: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error chunking text: {str(e)}"
        )
