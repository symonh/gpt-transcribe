"""
Background worker for processing transcription jobs using Redis Queue.
"""
import os
import redis
from rq import Worker, Queue, Connection

# Redis connection
redis_url = os.environ.get('REDIS_URL', 'redis://localhost:6379')
conn = redis.from_url(redis_url)

if __name__ == '__main__':
    with Connection(conn):
        worker = Worker(['transcription'])
        worker.work()

