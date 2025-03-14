from typing import List, Tuple, Dict, Optional, Set
import logging
import uuid
from io import BytesIO

from .models import (
    ChunkingResult,
    Chunk,
)

from sdpm import SDPMChunker

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
    try:
        # Get the conversion job result
        job_result = self.get_batch_conversion_task_result(job_id, include_page_numbers=True)
        
        # Check if job was successful
        if job_result.status != "SUCCESS" or not job_result.conversion_results:
            return ChunkingResult(
                job_id=job_id,
                filename="unknown",
                error=f"Failed to retrieve valid conversion result: {job_result.error or 'No conversion results found'}"
            )
        
        # Get the first conversion result (assuming single document per job)
        conversion_result = job_result.conversion_results[0].result
        filename = conversion_result.filename
        
        # Initialize the chunker with recommended settings
        chunker = SDPMChunker(
            embedding_model="minishlab/potion-base-8M",  # Default recommended model
            mode="window",                              # Mode for grouping sentences
            threshold="auto",                           # Auto-calculate similarity threshold
            chunk_size=max_tokens,                      # Maximum tokens per chunk
            similarity_window=1,                        # Number of sentences for similarity calculation
            min_sentences=1                             # Initial sentences per chunk
        )
        
        # Process the text through the chunker
        chunk_results = chunker(conversion_result.text)
        
        # Convert chunker results to our Chunk model and add page numbers
        chunks = []
        
        # If we need to include page numbers
        if include_page_numbers and conversion_result.page_content:
            # Map text positions to page numbers
            text_to_page_map = {}
            current_position = 0
            
            # Sort page content by page number
            sorted_pages = sorted(conversion_result.page_content.items(), key=lambda x: int(x[0]))
            
            for page_num, content in sorted_pages:
                page_length = len(content)
                # Map each character position to its page number
                for i in range(current_position, current_position + page_length):
                    text_to_page_map[i] = int(page_num)
                current_position += page_length
            
            # Now process each chunk and determine its page range
            for chunk_result in chunk_results:
                # Find the start position of this chunk in the full text
                start_pos = conversion_result.text.find(chunk_result.text)
                if start_pos == -1:
                    # If exact match not found (possible due to whitespace differences)
                    # use a more flexible approach or skip page numbering for this chunk
                    chunk_metadata = {
                        "token_count": str(chunk_result.token_count),
                        "sentence_count": str(len(chunk_result.sentences))
                    }
                    chunks.append(Chunk(
                        text=chunk_result.text,
                        metadata=chunk_metadata,
                        page_numbers=None,
                        start_page=None,
                        end_page=None
                    ))
                    continue
                
                end_pos = start_pos + len(chunk_result.text) - 1
                
                # Determine page range
                start_page = None
                end_page = None
                page_numbers = set()
                
                # Sample positions throughout the chunk to determine page coverage
                # This is more efficient than checking every position
                sampling_interval = max(1, len(chunk_result.text) // 10)
                for pos in range(start_pos, end_pos + 1, sampling_interval):
                    if pos in text_to_page_map:
                        page_num = text_to_page_map[pos]
                        page_numbers.add(page_num)
                        if start_page is None or page_num < start_page:
                            start_page = page_num
                        if end_page is None or page_num > end_page:
                            end_page = page_num
                
                # Also check the end position explicitly
                if end_pos in text_to_page_map:
                    page_num = text_to_page_map[end_pos]
                    page_numbers.add(page_num)
                    if end_page is None or page_num > end_page:
                        end_page = page_num
                
                chunk_metadata = {
                    "token_count": str(chunk_result.token_count),
                    "sentence_count": str(len(chunk_result.sentences))
                }
                
                chunks.append(Chunk(
                    text=chunk_result.text,
                    metadata=chunk_metadata,
                    page_numbers=sorted(list(page_numbers)) if page_numbers else None,
                    start_page=start_page,
                    end_page=end_page
                ))
        else:
            # Without page numbers, process chunks normally
            for chunk_result in chunk_results:
                chunk_metadata = {
                    "token_count": str(chunk_result.token_count),
                    "sentence_count": str(len(chunk_result.sentences))
                }
                
                chunks.append(Chunk(
                    text=chunk_result.text,
                    metadata=chunk_metadata,
                    page_numbers=None,
                    start_page=None,
                    end_page=None
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
            filename="unknown",
            error=f"Error during chunking: {str(e)}"
        ) 