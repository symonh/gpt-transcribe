"""
Background worker for processing transcription jobs using Redis Queue.
"""
import os
import sys
import logging
import redis
from rq import Worker, Queue, SimpleWorker

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Redis connection - Heroku Redis uses self-signed certs, need to disable verification
redis_url = os.environ.get('REDIS_URL', 'redis://localhost:6379')

# For Heroku Redis (rediss:// URLs), disable SSL verification
if redis_url.startswith('rediss://'):
    conn = redis.from_url(redis_url, ssl_cert_reqs=None)
else:
    conn = redis.from_url(redis_url)

if __name__ == '__main__':
    logger.info("Starting transcription worker...")
    queues = [Queue('transcription', connection=conn)]
    
    # Use SimpleWorker on macOS to avoid fork issues
    # SimpleWorker runs jobs in the main process (no forking)
    if sys.platform == 'darwin':
        logger.info("Using SimpleWorker (no fork) for macOS compatibility")
        worker = SimpleWorker(queues, connection=conn)
    else:
        worker = Worker(queues, connection=conn)
    
    worker.work()
