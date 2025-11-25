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
    Returns the transcription result.
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
        
        # Step 1: Transcribe with diarization model
        logger.info("Step 1: Transcribing with diarization model...")
        raw_transcript = transcribe_with_diarization(temp_file_path)
        logger.info(f"Raw transcript length: {len(raw_transcript)} chars")
        
        # Step 2: Use GPT-4 to identify speakers
        logger.info("Step 2: Identifying speakers with GPT-4...")
        diarized_result = identify_speakers_with_gpt4(raw_transcript)
        
        result = {
            'status': 'completed',
            'text': raw_transcript,
            'segments': diarized_result.get('segments', [])
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
    Use OpenAI's diarization model via REST API
    """
    with open(file_path, 'rb') as audio_file:
        response = requests.post(
            'https://api.openai.com/v1/audio/transcriptions',
            headers={'Authorization': f'Bearer {OPENAI_API_KEY}'},
            files={'file': audio_file},
            data={
                'model': 'gpt-4o-transcribe-diarize',
                'response_format': 'text',
                'chunking_strategy': 'auto'
            },
            timeout=600
        )
    
    if response.status_code != 200:
        raise Exception(f"Transcription API error: {response.text}")
    
    return response.text


def identify_speakers_with_gpt4(transcript_text):
    """
    Use GPT-4 to identify and label different speakers
    """
    system_prompt = """You are a transcript formatter. Your job is to take a meeting transcript and format it with clear speaker labels.

Rules:
1. Identify distinct speakers based on conversational patterns, context, and speaking styles
2. Label speakers as "Speaker 1", "Speaker 2", etc. (or use names if you can identify them from context)
3. Format the output as a JSON array of segments with this structure:
   {"segments": [{"speaker": "Speaker 1", "text": "what they said"}, ...]}
4. Merge consecutive segments from the same speaker
5. Keep the text faithful to the original - don't paraphrase
6. If there are interjections like "Mm-hmm", "Right", "Yeah" - attribute them to the appropriate speaker based on context

Return ONLY valid JSON, no other text."""

    user_prompt = f"""Here is a meeting transcript. Please identify the speakers and format it as JSON:

{transcript_text}"""

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        response_format={"type": "json_object"},
        temperature=0.1
    )
    
    return json.loads(response.choices[0].message.content)

