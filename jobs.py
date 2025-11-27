"""
Background job functions for transcription processing.
Uses OpenAI's gpt-4o-transcribe-diarize model which handles chunking internally.
"""
import os
import base64
import tempfile
import requests
import logging

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


def transcribe_audio_job(file_data_b64, filename):
    """
    Background job to transcribe audio file.
    Sends the file directly to OpenAI which handles chunking internally via chunking_strategy='auto'.
    
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
        
        logger.info(f"File saved to temp: {temp_file_path}, size: {file_size / 1024 / 1024:.2f} MB")
        
        # Transcribe with diarization model
        # OpenAI handles chunking internally with chunking_strategy='auto'
        logger.info("Sending to OpenAI for transcription (chunking handled by API)...")
        diarized_result = transcribe_with_diarization(temp_file_path)
        
        # Extract and merge consecutive segments from the same speaker
        raw_segments = diarized_result.get('segments', [])
        segments = merge_consecutive_speaker_segments(raw_segments)
        
        logger.info(f"Transcription completed: {len(raw_segments)} raw segments -> {len(segments)} merged segments")
        
        result = {
            'status': 'completed',
            'text': diarized_result.get('text', ''),
            'duration': diarized_result.get('duration', 0),
            'segments': segments
        }
        
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


def transcribe_with_diarization(file_path):
    """
    Use OpenAI's diarization model via REST API with diarized_json format.
    The API handles chunking internally via chunking_strategy='auto'.
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
            timeout=1800  # 30 minute timeout for large files
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
