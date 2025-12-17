"""
Background job functions for transcription processing.
Handles audio conversion, chunking for large files, and OpenAI transcription.
"""
import os
import base64
import tempfile
import subprocess
import requests
import logging
import math
import redis
from concurrent.futures import ThreadPoolExecutor, as_completed

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Configuration
MAX_FILE_SIZE_MB = 24  # OpenAI limit is 25MB, leave buffer
CHUNK_DURATION_MINUTES = 20  # Duration of each chunk for large files

_openai_api_key = None
_redis_conn = None

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


def get_redis_connection():
    """Get Redis connection, creating it if needed."""
    global _redis_conn
    if _redis_conn is None:
        try:
            import config
            redis_url = getattr(config, 'REDIS_URL', 'redis://localhost:6379')
        except ImportError:
            redis_url = os.environ.get('REDIS_URL', 'redis://localhost:6379')
        
        if redis_url.startswith('rediss://'):
            _redis_conn = redis.from_url(redis_url, ssl_cert_reqs=None)
        else:
            _redis_conn = redis.from_url(redis_url)
    return _redis_conn


def convert_to_mp3(input_path, output_path):
    """
    Convert audio file to mp3 format using ffmpeg.
    Uses 64kbps mono 16kHz which is optimal for speech recognition.
    """
    cmd = [
        'ffmpeg', '-y', '-i', input_path,
        '-vn',  # No video
        '-ar', '16000',  # 16kHz sample rate
        '-ac', '1',  # Mono
        '-b:a', '64k',  # 64kbps bitrate
        output_path
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise Exception(f"ffmpeg conversion failed: {result.stderr}")
    
    return output_path


def get_audio_duration(file_path):
    """Get audio duration in seconds using ffprobe."""
    cmd = [
        'ffprobe', '-v', 'error',
        '-show_entries', 'format=duration',
        '-of', 'default=noprint_wrappers=1:nokey=1',
        file_path
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise Exception(f"ffprobe failed: {result.stderr}")
    
    return float(result.stdout.strip())


def split_audio(input_path, chunk_duration_seconds, output_dir):
    """
    Split audio file into chunks of specified duration.
    Returns list of chunk file paths.
    """
    total_duration = get_audio_duration(input_path)
    num_chunks = math.ceil(total_duration / chunk_duration_seconds)
    
    chunk_paths = []
    for i in range(num_chunks):
        start_time = i * chunk_duration_seconds
        chunk_path = os.path.join(output_dir, f'chunk_{i:03d}.mp3')
        
        cmd = [
            'ffmpeg', '-y', '-i', input_path,
            '-ss', str(start_time),
            '-t', str(chunk_duration_seconds),
            '-vn', '-ar', '16000', '-ac', '1', '-b:a', '64k',
            chunk_path
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise Exception(f"ffmpeg split failed: {result.stderr}")
        
        # Only add if file has content
        if os.path.exists(chunk_path) and os.path.getsize(chunk_path) > 1000:
            chunk_paths.append(chunk_path)
    
    return chunk_paths


def transcribe_audio_job(file_key_or_data, filename, use_redis_key=True):
    """
    Background job to transcribe audio file.
    Handles conversion, chunking, and merging for large files.
    
    file_key_or_data: Redis key containing file data, or base64-encoded data if use_redis_key=False
    filename: Original filename (for extension)
    use_redis_key: If True, fetch file data from Redis using key. If False, treat as base64 data.
    Returns the transcription result with speaker diarization and timestamps.
    """
    temp_dir = None
    redis_key = None
    try:
        logger.info(f"Starting transcription job for: {filename}")
        
        # Create temp directory for all working files
        temp_dir = tempfile.mkdtemp(prefix='transcribe_')
        
        # Get file data either from Redis or from base64
        if use_redis_key:
            redis_key = file_key_or_data
            logger.info(f"Fetching file data from Redis key: {redis_key}")
            redis_conn = get_redis_connection()
            file_data = redis_conn.get(redis_key)
            if file_data is None:
                raise Exception(f"File data not found in Redis for key: {redis_key}")
            # Delete the key after fetching to free up memory
            redis_conn.delete(redis_key)
            logger.info("File data retrieved and Redis key deleted")
        else:
            logger.info("Using base64 encoded file data (sync mode)")
            file_data = base64.b64decode(file_key_or_data)
        
        file_size_mb = len(file_data) / (1024 * 1024)
        ext = os.path.splitext(filename)[1].lower()
        
        original_path = os.path.join(temp_dir, f'original{ext}')
        with open(original_path, 'wb') as f:
            f.write(file_data)
        
        logger.info(f"File saved: {file_size_mb:.2f} MB")
        
        # Convert to mp3 for compatibility
        logger.info("Converting to mp3...")
        mp3_path = os.path.join(temp_dir, 'converted.mp3')
        convert_to_mp3(original_path, mp3_path)
        
        mp3_size_mb = os.path.getsize(mp3_path) / (1024 * 1024)
        logger.info(f"Converted mp3 size: {mp3_size_mb:.2f} MB")
        
        # Get audio duration
        audio_duration = get_audio_duration(mp3_path)
        logger.info(f"Audio duration: {audio_duration:.2f} seconds ({audio_duration/60:.2f} minutes)")
        
        # gpt-4o-transcribe-diarize has a 1400 second (23.3 minute) limit
        MAX_DURATION_SECONDS = 1400
        
        # Check if we need to chunk based on duration OR file size
        if audio_duration > MAX_DURATION_SECONDS or mp3_size_mb > MAX_FILE_SIZE_MB:
            if audio_duration > MAX_DURATION_SECONDS:
                logger.info(f"Audio too long ({audio_duration/60:.1f} min > {MAX_DURATION_SECONDS/60:.1f} min), splitting into chunks...")
            else:
                logger.info(f"File too large ({mp3_size_mb:.1f}MB > {MAX_FILE_SIZE_MB}MB), splitting into chunks...")
            chunk_duration = CHUNK_DURATION_MINUTES * 60  # Convert to seconds
            chunk_paths = split_audio(mp3_path, chunk_duration, temp_dir)
            logger.info(f"Split into {len(chunk_paths)} chunks")
            
            # Get durations for each chunk (needed for timestamp offsets)
            chunk_durations = [get_audio_duration(cp) for cp in chunk_paths]
            
            # Calculate time offsets for each chunk
            time_offsets = [0]
            for i in range(len(chunk_durations) - 1):
                time_offsets.append(time_offsets[-1] + chunk_durations[i])
            
            # Transcribe ALL chunks in PARALLEL for speed
            logger.info(f"Transcribing {len(chunk_paths)} chunks in parallel...")
            chunk_results = [None] * len(chunk_paths)
            
            with ThreadPoolExecutor(max_workers=len(chunk_paths)) as executor:
                future_to_idx = {
                    executor.submit(transcribe_single_file, chunk_path): i 
                    for i, chunk_path in enumerate(chunk_paths)
                }
                
                for future in as_completed(future_to_idx):
                    idx = future_to_idx[future]
                    try:
                        chunk_results[idx] = future.result()
                        logger.info(f"Chunk {idx+1}/{len(chunk_paths)} completed")
                    except Exception as e:
                        logger.error(f"Chunk {idx+1} failed: {e}")
                        raise
            
            # Combine results with proper time offsets
            all_segments = []
            for i, chunk_result in enumerate(chunk_results):
                time_offset = time_offsets[i]
                for segment in chunk_result.get('segments', []):
                    segment['start'] = segment.get('start', 0) + time_offset
                    segment['end'] = segment.get('end', 0) + time_offset
                    all_segments.append(segment)
            
            # Merge consecutive speaker segments
            segments = merge_consecutive_speaker_segments(all_segments)
            full_text = ' '.join(seg.get('text', '') for seg in segments)
            total_duration = sum(chunk_durations)
            
        else:
            # Single file transcription
            logger.info("Transcribing file...")
            result = transcribe_single_file(mp3_path)
            
            raw_segments = result.get('segments', [])
            segments = merge_consecutive_speaker_segments(raw_segments)
            full_text = result.get('text', '')
            total_duration = result.get('duration', 0)
        
        logger.info(f"Transcription completed: {len(segments)} segments")
        
        return {
            'status': 'completed',
            'text': full_text,
            'duration': total_duration,
            'segments': segments
        }
    
    except Exception as e:
        logger.exception(f"Error in transcription job: {e}")
        return {
            'status': 'failed',
            'error': str(e)
        }
    
    finally:
        # Clean up temp directory
        if temp_dir and os.path.exists(temp_dir):
            import shutil
            shutil.rmtree(temp_dir, ignore_errors=True)
            logger.info("Temporary files cleaned up")


def transcribe_single_file(file_path):
    """
    Transcribe a single audio file using OpenAI's diarization model.
    Uses gpt-4o-transcribe-diarize with diarized_json format.
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
            timeout=600  # 10 minute timeout per chunk
        )
    
    if response.status_code != 200:
        logger.error(f"OpenAI API error response: {response.text}")
        raise Exception(f"Transcription API error: {response.text}")
    
    result = response.json()
    
    # Parse the response - should include segments with speaker labels
    segments = []
    for seg in result.get('segments', []):
        segments.append({
            'speaker': seg.get('speaker', 'Speaker'),
            'text': seg.get('text', '').strip(),
            'start': seg.get('start', 0),
            'end': seg.get('end', 0)
        })
    
    return {
        'text': result.get('text', ''),
        'duration': result.get('duration', 0),
        'segments': segments
    }


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
