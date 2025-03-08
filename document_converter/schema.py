from pydantic import BaseModel, Field
from typing import List, Literal, Optional, Dict, Any, Union
import base64


class ImageData(BaseModel):
    content: str = Field(..., description="Base64 encoded image content")
    format: str = Field(..., description="Image format (e.g., 'png', 'jpg')")


class ConversionResult(BaseModel):
    filename: str = Field(..., description="Original filename of the document")
    markdown: Optional[str] = Field(None, description="Converted markdown content")
    images: List[ImageData] = Field(default_factory=list, description="Images extracted from the document")
    error: Optional[str] = Field(None, description="Error message if conversion failed")


class BatchConversionResult(BaseModel):
    conversion_results: List[ConversionResult] = Field(
        default_factory=list, description="The results of the conversions"
    )


class ConversionJobResult(BaseModel):
    job_id: str = Field(..., description="The id of the conversion job")
    status: str = Field(..., description="Current status of the job")
    result: Optional[ConversionResult] = Field(None, description="The conversion result")
    error: Optional[str] = Field(None, description="Error message if job failed")


class BatchConversionJobResult(BaseModel):
    job_id: str = Field(..., description="The id of the batch conversion job")
    status: str = Field(..., description="Current status of the batch job")
    conversion_results: List[ConversionJobResult] = Field(default_factory=list, description="Individual conversion job results")
    error: Optional[str] = Field(None, description="Error message if batch job failed")


class Chunk(BaseModel):
    text: str = Field(..., description="The plain text content of the chunk")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata associated with the chunk")


class ChunkingResult(BaseModel):
    job_id: str = Field(..., description="The id of the original conversion job")
    filename: str = Field(..., description="The filename of the document")
    chunks: List[Chunk] = Field(default_factory=list, description="The chunks extracted from the document")
    error: Optional[str] = Field(None, description="The error that occurred during chunking")


class TextChunkingRequest(BaseModel):
    text: str = Field(..., description="The text content to chunk")
    filename: Optional[str] = Field("input.txt", description="A name to identify the source (for reporting purposes)")
    max_tokens: int = Field(512, ge=64, le=2048, description="Maximum number of tokens per chunk")
    merge_peers: bool = Field(True, description="Whether to merge undersized peer chunks")
    include_page_numbers: bool = Field(False, description="Whether to include page number references in chunk metadata")
    
    class Config:
        schema_extra = {
            "example": {
                "text": "This is the text content that needs to be chunked. It can be as long as needed.",
                "filename": "example.txt",
                "max_tokens": 512,
                "merge_peers": True,
                "include_page_numbers": False
            }
        }