"""
Background job functions for transcription processing.
"""
import os
import json
import base64
import tempfile
import requests
import logging
from openai import OpenAI

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Configuration
try:
    import config
    OPENAI_API_KEY = config.OPENAI_API_KEY
except ImportError:
    OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')

# Initialize OpenAI client
client = OpenAI(api_key=OPENAI_API_KEY)


def transcribe_audio_job(file_data_b64, filename):
    """
    Background job to transcribe audio file.
    file_data_b64: Base64 encoded file content
    filename: Original filename (for extension)
    Returns the transcription result with speaker diarization and timestamps.
    """
    temp_file_path = None
    try:
        logger.info(f"Starting transcription job for: {filename}")
        
        # Decode base64 file data and save to temp file
        file_data = base64.b64decode(file_data_b64)
        ext = os.path.splitext(filename)[1]
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as temp_file:
            temp_file.write(file_data)
            temp_file_path = temp_file.name
        
        logger.info(f"File saved to worker temp: {temp_file_path}, size: {len(file_data)} bytes")
        
        # Transcribe with diarization model using diarized_json format
        # This gives us speaker labels AND timestamps in one API call
        logger.info("Transcribing with diarization model (diarized_json format)...")
        diarized_result = transcribe_with_diarization(temp_file_path)
        
        # Extract segments with timestamps
        segments = []
        for seg in diarized_result.get('segments', []):
            segments.append({
                'speaker': seg.get('speaker', 'Speaker'),
                'text': seg.get('text', ''),
                'start': seg.get('start', 0),
                'end': seg.get('end', 0),
                'id': seg.get('id', '')
            })
        
        result = {
            'status': 'completed',
            'text': diarized_result.get('text', ''),
            'duration': diarized_result.get('duration', 0),
            'segments': segments
        }
        
        logger.info(f"Transcription completed with {len(result['segments'])} segments")
        
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
    Returns segments with speaker labels, text, and timestamps (start/end).
    """
    with open(file_path, 'rb') as audio_file:
        response = requests.post(
            'https://api.openai.com/v1/audio/transcriptions',
            headers={'Authorization': f'Bearer {OPENAI_API_KEY}'},
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

