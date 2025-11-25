"""
Configuration file for GPT Transcribe application.
Copy this file to config.py and fill in your credentials.

IMPORTANT: Never commit config.py to version control!
"""

import os

# OpenAI Configuration
# Get your API key from https://platform.openai.com/api-keys
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY', 'your-openai-api-key-here')

# Gmail Configuration for sending transcripts
# Use a Gmail App Password: https://support.google.com/accounts/answer/185833
GMAIL_SENDER_EMAIL = os.environ.get('GMAIL_SENDER_EMAIL', 'your-email@gmail.com')
GMAIL_APP_PASSWORD = os.environ.get('GMAIL_APP_PASSWORD', 'your-app-password-here')

# App Configuration
MAX_CONTENT_LENGTH = 100 * 1024 * 1024  # 100MB max file size
ALLOWED_EXTENSIONS = {'mp3', 'mp4', 'mpeg', 'mpga', 'm4a', 'wav', 'webm'}

