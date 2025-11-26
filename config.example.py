"""
Configuration file for GPT Transcribe application.
Copy this file to config.py and fill in your credentials.

IMPORTANT: Never commit config.py to version control!

To generate a password hash, run:
    python -c "from werkzeug.security import generate_password_hash; print(generate_password_hash('your-password'))"
"""

import os

# OpenAI Configuration
# Get your API key from https://platform.openai.com/api-keys
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY', 'your-openai-api-key-here')

# Gmail Configuration for sending transcripts
# Use a Gmail App Password: https://support.google.com/accounts/answer/185833
GMAIL_SENDER_EMAIL = os.environ.get('GMAIL_SENDER_EMAIL', 'your-email@gmail.com')
GMAIL_APP_PASSWORD = os.environ.get('GMAIL_APP_PASSWORD', 'your-app-password-here')

# Authentication Configuration
APP_USERNAME = 'admin'
APP_PASSWORD_HASH = 'your-password-hash-here'  # Generate with werkzeug.security.generate_password_hash()
SECRET_KEY = 'generate-a-random-secret-key'  # Use: python -c "import secrets; print(secrets.token_hex(32))"

# App Configuration
MAX_CONTENT_LENGTH = 300 * 1024 * 1024  # 300MB max file size
ALLOWED_EXTENSIONS = {'mp3', 'mp4', 'mpeg', 'mpga', 'm4a', 'wav', 'webm'}

# Parallel Processing - number of chunks to transcribe simultaneously
MAX_PARALLEL_CHUNKS = 10  # Higher = faster for long files, but uses more API concurrency

