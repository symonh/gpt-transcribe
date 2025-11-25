# Meeting Transcriber - AI-Powered Diarization

A simple yet attractive web application that transcribes audio recordings of meetings into diarized transcripts with speaker identification using OpenAI's GPT-4o models.

## Features

- 👥 **Speaker Diarization**: Automatically identifies and labels different speakers (Speaker 1, Speaker 2, etc.)
- 📝 **Accurate Transcription**: Two-step process using GPT-4o transcription + GPT-4 speaker identification
- 🎨 **Visual Speaker Distinction**: Color-coded speaker badges and borders for easy reading
- 🎨 **Modern UI**: Beautiful, responsive interface with drag-and-drop support
- ⚡ **Fast Processing**: Efficient audio processing pipeline
- 📋 **Easy Export**: Copy transcripts to clipboard with one click

## Supported Audio Formats

- MP3
- MP4
- MPEG
- MPGA
- M4A
- WAV
- WebM

Maximum file size: 100MB

## Local Development

### Prerequisites

- Python 3.11+
- OpenAI API key with access to GPT-4o Transcribe Diarize model

### Setup

1. Clone the repository:
```bash
git clone <your-repo-url>
cd gpt-transcribe
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set your OpenAI API key:
```bash
export OPENAI_API_KEY='your-api-key-here'
```

**Note:** The API key is included in the code by default for convenience, but for production deployment, you should use environment variables.

5. Run the application:
```bash
python app.py
```

Or specify a custom port:
```bash
python app.py --port 8080
```

6. Open your browser and navigate to:
```
http://localhost:5001
```

**Note:** 
- Default port is 5001 (to avoid conflicts with macOS AirPlay Receiver on port 5000)
- You can override the port using the `--port` flag
- The `PORT` environment variable also works (useful for Heroku deployment)
- Priority: `--port` flag > `PORT` env variable > default (5001)

## Deployment to Heroku

1. Install the Heroku CLI from https://devcenter.heroku.com/articles/heroku-cli

2. Login to Heroku:
```bash
heroku login
```

3. Create a new Heroku app:
```bash
heroku create your-app-name
```

4. Set the OpenAI API key as an environment variable:
```bash
heroku config:set OPENAI_API_KEY='your-api-key-here'
```

5. Deploy to Heroku:
```bash
git add .
git commit -m "Initial deployment"
git push heroku main
```

6. Open your app:
```bash
heroku open
```

## Usage

1. Open the web application
2. Click or drag-and-drop an audio file into the upload area
3. Wait for the transcription to complete (processing time depends on audio length)
4. View the diarized transcript with speaker labels and timestamps
5. Copy the transcript to clipboard if needed

### Testing Tips

- Use clear audio recordings with minimal background noise for best results
- Ensure speakers are distinct and speak clearly
- Supported file formats: MP3, MP4, MPEG, MPGA, M4A, WAV, WebM
- Maximum file size: 100MB
- For testing, you can use sample meeting recordings or create a short audio file with multiple speakers

## Security Notes

- The API key is stored as an environment variable for security
- Temporary audio files are automatically deleted after processing
- Maximum file size limits prevent resource exhaustion

## Technologies Used

- **Backend**: Flask (Python)
- **AI Model**: OpenAI GPT-4o Transcribe Diarize
- **Frontend**: HTML, CSS, JavaScript
- **Deployment**: Heroku
- **WSGI Server**: Gunicorn

### Important Dependencies

The project uses specific versions of `httpx`, `httpcore`, and `h11` to ensure compatibility with the OpenAI client library:
- `httpx==0.24.1`
- `httpcore==0.17.3`
- `h11==0.14.0`

These versions are locked in `requirements.txt` to prevent compatibility issues during deployment.

## License

MIT License

