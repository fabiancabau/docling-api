from pydantic import BaseModel, Field
from typing import List, Literal, Optional, Dict, Any, Union
import base64
from enum import Enum


class ImageData(BaseModel):
    type: str = Field(..., description="Type of image (e.g., 'table', 'picture')")
    filename: str = Field(..., description="Filename of the image")
    image: str = Field(..., description="Base64 encoded image content")


class ConversionResult(BaseModel):
    filename: str = Field(..., description="Original filename of the document")
    markdown: Optional[str] = Field(None, description="Converted markdown content")
    images: Optional[List[ImageData]] = Field(None, description="Images extracted from the document")
    error: Optional[str] = Field(None, description="Error message if conversion failed")
    page_content: Optional[Dict[str, Optional[str]]] = Field(None, description="Markdown content organized by page number")


class BatchConversionResult(BaseModel):
    conversion_results: Optional[List[ConversionResult]] = Field(
        None, description="The results of the conversions"
    )


class TaskStatus(str, Enum):
    SUCCESS = "SUCCESS"
    IN_PROGRESS = "IN_PROGRESS"
    FAILURE = "FAILURE"
    PENDING = "PENDING"

class ConversionJobResult(BaseModel):
    job_id: str = Field(..., description="The id of the conversion job")
    status: TaskStatus = Field(..., description="Current status of the job")
    result: Optional[ConversionResult] = Field(None, description="The conversion result, present when status is SUCCESS")
    error: Optional[str] = Field(None, description="Error message if job failed, present when status is FAILURE")


class BatchConversionJobResult(BaseModel):
    job_id: str = Field(..., description="The id of the batch conversion job")
    status: TaskStatus = Field(..., description="Current status of the batch job")
    conversion_results: Optional[List[ConversionJobResult]] = Field(None, description="Individual conversion job results, present when status is SUCCESS")
    error: Optional[str] = Field(None, description="Error message if batch job failed, present when status is FAILURE")


class Chunk(BaseModel):
    text: str = Field(..., description="The plain text content of the chunk")
    metadata: Optional[Dict[str, str]] = Field(None, description="Additional metadata including token_count and sentence_count")
    page_numbers: Optional[List[int]] = Field(None, description="List of page numbers this chunk spans across, only present if include_page_numbers is True")
    start_page: Optional[int] = Field(None, description="The page number where this chunk starts, only present if include_page_numbers is True")
    end_page: Optional[int] = Field(None, description="The page number where this chunk ends, only present if include_page_numbers is True")


class ChunkingStatus(str, Enum):
    SUCCESS = "SUCCESS"
    FAILURE = "FAILURE"


class ChunkingResult(BaseModel):
    job_id: str = Field(..., description="The id of the original conversion job")
    filename: str = Field(..., description="The filename of the document")
    chunks: List[Chunk] = Field(default_factory=list, description="The chunks extracted from the document")
    error: Optional[str] = Field(None, description="Error message if chunking failed")


class TextChunkingRequest(BaseModel):
    text: str = Field(..., description="The text content to chunk")
    filename: str = Field(default="input.txt", description="A name to identify the source (for reporting purposes)")
    max_tokens: int = Field(default=512, description="Maximum number of tokens per chunk")
    merge_peers: bool = Field(default=True, description="Whether to merge undersized peer chunks")
    include_page_numbers: bool = Field(default=True, description="Whether to include page number references in chunk metadata")


class HealthCheckResponse(BaseModel):
    status: str = Field(..., description="Overall health status")
    services: Optional[Dict[str, str]] = Field(None, description="Status of individual services")