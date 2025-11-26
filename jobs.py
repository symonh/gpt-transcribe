"""
Background job functions for transcription processing.
Supports chunking large audio files and processing them in parallel for faster transcription.
"""
import os
import json
import base64
import tempfile
import requests
import logging
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Configuration - API key loaded lazily for testing without key
_openai_api_key = None

def get_openai_api_key():
    """Get the OpenAI API key, loading it only when needed."""
    global _openai_api_key
    if _openai_api_key is None:
        try:
            import config
            _openai_api_key = config.OPENAI_API_KEY
        except ImportError:
            _openai_api_key = os.environ.get('OPENAI_API_KEY')
    return _openai_api_key

# Chunk configuration
CHUNK_DURATION_MS = 10 * 60 * 1000  # 10 minutes per chunk
CHUNK_SIZE_THRESHOLD = 25 * 1024 * 1024  # Only chunk files larger than 25MB

# Max parallel chunks - configurable via environment or config
try:
    import config
    MAX_PARALLEL_CHUNKS = getattr(config, 'MAX_PARALLEL_CHUNKS', 10)
except ImportError:
    MAX_PARALLEL_CHUNKS = int(os.environ.get('MAX_PARALLEL_CHUNKS', '10'))


def transcribe_audio_job(file_data_b64, filename):
    """
    Background job to transcribe audio file.
    For large files, automatically splits into chunks and processes in parallel.
    
    file_data_b64: Base64 encoded file content
    filename: Original filename (for extension)
    Returns the transcription result with speaker diarization and timestamps.
    """
    temp_file_path = None
    try:
        logger.info(f"Starting transcription job for: {filename}")
        
        # Decode base64 file data and save to temp file
        file_data = base64.b64decode(file_data_b64)
        file_size = len(file_data)
        ext = os.path.splitext(filename)[1].lower()
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as temp_file:
            temp_file.write(file_data)
            temp_file_path = temp_file.name
        
        logger.info(f"File saved to worker temp: {temp_file_path}, size: {file_size} bytes")
        
        # Determine if we should use chunking (for large files)
        if file_size > CHUNK_SIZE_THRESHOLD:
            logger.info(f"Large file detected ({file_size} bytes), using chunked parallel processing")
            result = transcribe_with_chunking(temp_file_path, ext)
        else:
            logger.info(f"Small file ({file_size} bytes), using single transcription")
            # Transcribe with diarization model using diarized_json format
            logger.info("Transcribing with diarization model (diarized_json format)...")
            diarized_result = transcribe_with_diarization(temp_file_path)
            
            # Extract and merge consecutive segments from the same speaker
            raw_segments = diarized_result.get('segments', [])
            segments = merge_consecutive_speaker_segments(raw_segments)
            
            logger.info(f"Merged {len(raw_segments)} raw segments into {len(segments)} merged segments")
            
            result = {
                'status': 'completed',
                'text': diarized_result.get('text', ''),
                'duration': diarized_result.get('duration', 0),
                'segments': segments
            }
        
        logger.info(f"Transcription completed with {len(result.get('segments', []))} segments")
        
        return result
    
    except Exception as e:
        logger.exception(f"Error in transcription job: {e}")
        return {
            'status': 'failed',
            'error': str(e)
        }
    
    finally:
        # Clean up temp file
        if temp_file_path and os.path.exists(temp_file_path):
            os.unlink(temp_file_path)
            logger.info("Temporary file cleaned up")


def transcribe_with_chunking(file_path, file_ext):
    """
    Split audio into chunks, transcribe in parallel, and merge results.
    Returns the combined transcription result.
    """
    chunk_paths = []
    try:
        # Try to import pydub for audio chunking
        try:
            from pydub import AudioSegment
        except ImportError:
            logger.warning("pydub not available, falling back to single transcription")
            diarized_result = transcribe_with_diarization(file_path)
            raw_segments = diarized_result.get('segments', [])
            segments = merge_consecutive_speaker_segments(raw_segments)
            return {
                'status': 'completed',
                'text': diarized_result.get('text', ''),
                'duration': diarized_result.get('duration', 0),
                'segments': segments
            }
        
        # Load audio file
        logger.info(f"Loading audio file for chunking: {file_path}")
        try:
            audio = AudioSegment.from_file(file_path)
        except Exception as e:
            logger.warning(f"Failed to load audio with pydub: {e}, falling back to single transcription")
            diarized_result = transcribe_with_diarization(file_path)
            raw_segments = diarized_result.get('segments', [])
            segments = merge_consecutive_speaker_segments(raw_segments)
            return {
                'status': 'completed',
                'text': diarized_result.get('text', ''),
                'duration': diarized_result.get('duration', 0),
                'segments': segments
            }
        
        duration_ms = len(audio)
        duration_seconds = duration_ms / 1000.0
        logger.info(f"Audio duration: {duration_seconds:.2f} seconds ({duration_ms}ms)")
        
        # Calculate number of chunks
        num_chunks = (duration_ms + CHUNK_DURATION_MS - 1) // CHUNK_DURATION_MS
        logger.info(f"Splitting into {num_chunks} chunks of ~{CHUNK_DURATION_MS // 1000 // 60} minutes each")
        
        # If only one chunk, no need to split
        if num_chunks <= 1:
            logger.info("Only one chunk needed, using single transcription")
            diarized_result = transcribe_with_diarization(file_path)
            raw_segments = diarized_result.get('segments', [])
            segments = merge_consecutive_speaker_segments(raw_segments)
            return {
                'status': 'completed',
                'text': diarized_result.get('text', ''),
                'duration': diarized_result.get('duration', 0),
                'segments': segments
            }
        
        # Create chunks
        chunks_info = []
        for i in range(num_chunks):
            start_ms = i * CHUNK_DURATION_MS
            end_ms = min((i + 1) * CHUNK_DURATION_MS, duration_ms)
            
            chunk_audio = audio[start_ms:end_ms]
            
            # Save chunk to temp file
            chunk_path = tempfile.NamedTemporaryFile(delete=False, suffix='.mp3').name
            chunk_audio.export(chunk_path, format='mp3')
            chunk_paths.append(chunk_path)
            
            chunks_info.append({
                'index': i,
                'path': chunk_path,
                'start_offset_seconds': start_ms / 1000.0,
                'duration_ms': end_ms - start_ms
            })
            
            logger.info(f"Created chunk {i + 1}/{num_chunks}: {start_ms}ms - {end_ms}ms")
        
        # Transcribe chunks in parallel
        logger.info(f"Starting parallel transcription of {len(chunks_info)} chunks with {MAX_PARALLEL_CHUNKS} workers")
        chunk_results = transcribe_chunks_parallel(chunks_info)
        
        # Merge results
        logger.info("Merging chunk results...")
        merged_result = merge_chunk_results(chunk_results, duration_seconds)
        
        return merged_result
        
    finally:
        # Clean up chunk temp files
        for chunk_path in chunk_paths:
            try:
                if os.path.exists(chunk_path):
                    os.unlink(chunk_path)
            except Exception as e:
                logger.warning(f"Failed to clean up chunk file {chunk_path}: {e}")


def transcribe_chunks_parallel(chunks_info):
    """
    Transcribe multiple chunks in parallel using ThreadPoolExecutor.
    Returns list of (chunk_info, result) tuples sorted by chunk index.
    """
    results = []
    lock = threading.Lock()
    
    def transcribe_chunk(chunk_info):
        """Transcribe a single chunk."""
        try:
            logger.info(f"Transcribing chunk {chunk_info['index'] + 1}...")
            result = transcribe_with_diarization(chunk_info['path'])
            logger.info(f"Chunk {chunk_info['index'] + 1} completed")
            return (chunk_info, result)
        except Exception as e:
            logger.error(f"Error transcribing chunk {chunk_info['index']}: {e}")
            return (chunk_info, {'error': str(e), 'segments': [], 'text': ''})
    
    with ThreadPoolExecutor(max_workers=MAX_PARALLEL_CHUNKS) as executor:
        futures = {executor.submit(transcribe_chunk, chunk): chunk for chunk in chunks_info}
        
        for future in as_completed(futures):
            chunk_info, result = future.result()
            with lock:
                results.append((chunk_info, result))
    
    # Sort by chunk index to maintain order
    results.sort(key=lambda x: x[0]['index'])
    
    return results


def merge_chunk_results(chunk_results, total_duration):
    """
    Merge results from multiple chunks into a single transcript.
    Adjusts timestamps based on chunk offsets and normalizes speaker labels.
    """
    all_segments = []
    all_text_parts = []
    speaker_mapping = {}  # Maps (chunk_index, original_speaker) -> normalized_speaker
    next_speaker_id = 1
    
    for chunk_info, result in chunk_results:
        if 'error' in result:
            logger.warning(f"Chunk {chunk_info['index']} had error: {result['error']}")
            continue
        
        chunk_offset = chunk_info['start_offset_seconds']
        chunk_index = chunk_info['index']
        
        # Add text
        chunk_text = result.get('text', '')
        if chunk_text:
            all_text_parts.append(chunk_text)
        
        # Process segments with timestamp adjustment
        raw_segments = result.get('segments', [])
        
        for seg in raw_segments:
            original_speaker = seg.get('speaker', 'Speaker')
            text = seg.get('text', '').strip()
            start = seg.get('start', 0) + chunk_offset
            end = seg.get('end', 0) + chunk_offset
            
            if not text:
                continue
            
            # Map speaker labels: try to maintain consistency across chunks
            # For first chunk, create initial mappings
            # For subsequent chunks, try to match or create new speakers
            speaker_key = (chunk_index, original_speaker)
            
            if speaker_key not in speaker_mapping:
                # For simplicity, we maintain chunk-local speaker IDs
                # A more sophisticated approach would try to match voices across chunks
                # but that requires voice fingerprinting which is beyond current API capabilities
                normalized_speaker = f"Speaker {next_speaker_id}"
                
                # Try to preserve original speaker number if it exists
                try:
                    orig_num = int(original_speaker.split()[-1])
                    if orig_num <= 10:  # Reasonable speaker count
                        normalized_speaker = f"Speaker {orig_num}"
                except (ValueError, IndexError):
                    next_speaker_id += 1
                
                speaker_mapping[speaker_key] = normalized_speaker
            
            all_segments.append({
                'speaker': speaker_mapping[speaker_key],
                'text': text,
                'start': start,
                'end': end
            })
    
    # Sort segments by start time (should already be sorted, but ensure it)
    all_segments.sort(key=lambda x: x['start'])
    
    # Merge consecutive segments from the same speaker
    merged_segments = merge_consecutive_speaker_segments(all_segments)
    
    # Combine all text
    full_text = ' '.join(all_text_parts)
    
    logger.info(f"Merged {len(all_segments)} segments into {len(merged_segments)} final segments")
    
    return {
        'status': 'completed',
        'text': full_text,
        'duration': total_duration,
        'segments': merged_segments
    }


def transcribe_with_diarization(file_path):
    """
    Use OpenAI's diarization model via REST API with diarized_json format.
    Returns segments with speaker labels, text, and timestamps (start/end).
    """
    api_key = get_openai_api_key()
    if not api_key:
        raise Exception("OpenAI API key not configured")
    
    with open(file_path, 'rb') as audio_file:
        response = requests.post(
            'https://api.openai.com/v1/audio/transcriptions',
            headers={'Authorization': f'Bearer {api_key}'},
            files={'file': audio_file},
            data={
                'model': 'gpt-4o-transcribe-diarize',
                'response_format': 'diarized_json',
                'chunking_strategy': 'auto'
            },
            timeout=600
        )
    
    if response.status_code != 200:
        raise Exception(f"Transcription API error: {response.text}")
    
    return response.json()


def merge_consecutive_speaker_segments(raw_segments):
    """
    Merge consecutive segments from the same speaker into single segments.
    This prevents fragmented output where each word is a separate segment.
    """
    if not raw_segments:
        return []
    
    merged = []
    current_segment = None
    
    for seg in raw_segments:
        speaker = seg.get('speaker', 'Speaker')
        text = seg.get('text', '').strip()
        start = seg.get('start', 0)
        end = seg.get('end', 0)
        
        # Skip empty segments
        if not text:
            continue
        
        if current_segment is None:
            # Start a new segment
            current_segment = {
                'speaker': speaker,
                'text': text,
                'start': start,
                'end': end
            }
        elif current_segment['speaker'] == speaker:
            # Same speaker - merge by appending text and extending end time
            # Add space between text fragments
            current_segment['text'] += ' ' + text
            current_segment['end'] = end
        else:
            # Different speaker - save current and start new
            merged.append(current_segment)
            current_segment = {
                'speaker': speaker,
                'text': text,
                'start': start,
                'end': end
            }
    
    # Don't forget the last segment
    if current_segment is not None:
        merged.append(current_segment)
    
    # Add IDs to merged segments
    for i, seg in enumerate(merged):
        seg['id'] = f'seg_{i:03d}'
    
    return merged
