"""
Background worker for processing transcription jobs using Redis Queue.
"""
import os
import ssl
import redis
from rq import Worker, Queue, Connection

# Redis connection - Heroku Redis uses self-signed certs, need to disable verification
redis_url = os.environ.get('REDIS_URL', 'redis://localhost:6379')

# For Heroku Redis (rediss:// URLs), disable SSL verification
if redis_url.startswith('rediss://'):
    conn = redis.from_url(redis_url, ssl_cert_reqs=None)
else:
    conn = redis.from_url(redis_url)

if __name__ == '__main__':
    with Connection(conn):
        worker = Worker(['transcription'])
        worker.work()

