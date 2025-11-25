import os
import argparse
import logging
import requests
import json
import smtplib
import tempfile
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.application import MIMEApplication
from flask import Flask, request, jsonify, render_template, Response
from openai import OpenAI
from werkzeug.utils import secure_filename
from io import BytesIO
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.enums import TA_LEFT
import markdown as md

# Configuration - try to import config.py, fall back to environment variables
try:
    import config
    OPENAI_API_KEY = config.OPENAI_API_KEY
    GMAIL_SENDER_EMAIL = config.GMAIL_SENDER_EMAIL
    GMAIL_APP_PASSWORD = config.GMAIL_APP_PASSWORD
    MAX_CONTENT_LENGTH = config.MAX_CONTENT_LENGTH
    ALLOWED_EXTENSIONS = config.ALLOWED_EXTENSIONS
except ImportError:
    # Fall back to environment variables (for Heroku deployment)
    OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')
    GMAIL_SENDER_EMAIL = os.environ.get('GMAIL_SENDER_EMAIL')
    GMAIL_APP_PASSWORD = os.environ.get('GMAIL_APP_PASSWORD')
    MAX_CONTENT_LENGTH = 100 * 1024 * 1024  # 100MB
    ALLOWED_EXTENSIONS = {'mp3', 'mp4', 'mpeg', 'mpga', 'm4a', 'wav', 'webm'}

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH

# Initialize OpenAI client
client = OpenAI(api_key=OPENAI_API_KEY)


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def transcribe_with_diarization(file_path):
    """
    Use OpenAI's diarization model via REST API (since Python SDK doesn't support chunking_strategy yet)
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
            timeout=600  # 10 minute timeout for long files
        )
    
    if response.status_code != 200:
        raise Exception(f"Transcription API error: {response.text}")
    
    return response.text


def identify_speakers_with_gpt4(transcript_text):
    """
    Use GPT-4 to identify and label different speakers in the transcript
    """
    logger.info("Identifying speakers with GPT-4...")
    
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
    
    result = json.loads(response.choices[0].message.content)
    return result


def generate_pdf(segments, title="Meeting Transcript"):
    """Generate a nicely formatted PDF from transcript segments"""
    buffer = BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        rightMargin=0.75*inch,
        leftMargin=0.75*inch,
        topMargin=0.75*inch,
        bottomMargin=0.75*inch
    )
    
    styles = getSampleStyleSheet()
    
    # Custom styles
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        spaceAfter=30,
        textColor=colors.HexColor('#667eea'),
        fontName='Helvetica-Bold'
    )
    
    speaker_style = ParagraphStyle(
        'SpeakerName',
        parent=styles['Normal'],
        fontSize=11,
        textColor=colors.HexColor('#667eea'),
        fontName='Helvetica-Bold',
        spaceBefore=12,
        spaceAfter=4
    )
    
    text_style = ParagraphStyle(
        'SpeakerText',
        parent=styles['Normal'],
        fontSize=10,
        leading=14,
        textColor=colors.HexColor('#333333'),
        fontName='Helvetica',
        leftIndent=15,
        spaceAfter=8
    )
    
    story = []
    
    # Title
    story.append(Paragraph(title, title_style))
    story.append(Spacer(1, 20))
    
    # Speaker colors for visual distinction
    speaker_colors = [
        '#667eea', '#f5576c', '#4facfe', '#43e97b',
        '#fa709a', '#fee140', '#30cfd0', '#a8edea'
    ]
    speaker_color_map = {}
    color_index = 0
    
    for segment in segments:
        speaker = segment.get('speaker', 'Speaker')
        text = segment.get('text', '')
        
        # Assign color to speaker
        if speaker not in speaker_color_map:
            speaker_color_map[speaker] = speaker_colors[color_index % len(speaker_colors)]
            color_index += 1
        
        color = speaker_color_map[speaker]
        
        # Speaker label with color
        speaker_para_style = ParagraphStyle(
            f'Speaker_{speaker}',
            parent=speaker_style,
            textColor=colors.HexColor(color)
        )
        story.append(Paragraph(f"<b>{speaker}</b>", speaker_para_style))
        
        # Text content
        story.append(Paragraph(text, text_style))
    
    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()


def generate_text(segments):
    """Generate plain text format from transcript segments"""
    lines = []
    for segment in segments:
        speaker = segment.get('speaker', 'Speaker')
        text = segment.get('text', '')
        lines.append(f"{speaker}:\n{text}\n")
    return '\n'.join(lines)


def generate_markdown(segments, title="Meeting Transcript"):
    """Generate markdown format from transcript segments"""
    lines = [f"# {title}\n"]
    
    for segment in segments:
        speaker = segment.get('speaker', 'Speaker')
        text = segment.get('text', '')
        lines.append(f"**{speaker}:**\n\n{text}\n")
    
    return '\n'.join(lines)


def generate_html(segments, title="Meeting Transcript"):
    """Generate standalone HTML format from transcript segments"""
    speaker_colors = [
        '#667eea', '#f5576c', '#4facfe', '#43e97b',
        '#fa709a', '#fee140', '#30cfd0', '#a8edea'
    ]
    speaker_color_map = {}
    color_index = 0
    
    html_parts = [f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
            max-width: 800px;
            margin: 0 auto;
            padding: 40px 20px;
            background: #f5f5f5;
            color: #333;
        }}
        h1 {{
            color: #667eea;
            border-bottom: 3px solid #667eea;
            padding-bottom: 10px;
            margin-bottom: 30px;
        }}
        .segment {{
            background: white;
            border-radius: 8px;
            padding: 15px 20px;
            margin-bottom: 15px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .speaker {{
            font-weight: bold;
            margin-bottom: 8px;
            font-size: 14px;
        }}
        .text {{
            line-height: 1.6;
            color: #444;
        }}
    </style>
</head>
<body>
    <h1>{title}</h1>
"""]
    
    for segment in segments:
        speaker = segment.get('speaker', 'Speaker')
        text = segment.get('text', '')
        
        if speaker not in speaker_color_map:
            speaker_color_map[speaker] = speaker_colors[color_index % len(speaker_colors)]
            color_index += 1
        
        color = speaker_color_map[speaker]
        
        html_parts.append(f"""    <div class="segment">
        <div class="speaker" style="color: {color};">{speaker}</div>
        <div class="text">{text}</div>
    </div>
""")
    
    html_parts.append("""</body>
</html>""")
    
    return ''.join(html_parts)


def send_email(to_email, subject, body_text, body_html=None, attachment=None, attachment_name=None):
    """Send email using Gmail SMTP with app password"""
    msg = MIMEMultipart('alternative')
    msg['Subject'] = subject
    msg['From'] = GMAIL_SENDER_EMAIL
    msg['To'] = to_email
    
    # Add text body
    part1 = MIMEText(body_text, 'plain')
    msg.attach(part1)
    
    # Add HTML body if provided
    if body_html:
        part2 = MIMEText(body_html, 'html')
        msg.attach(part2)
    
    # Add attachment if provided
    if attachment and attachment_name:
        part = MIMEApplication(attachment, Name=attachment_name)
        part['Content-Disposition'] = f'attachment; filename="{attachment_name}"'
        msg.attach(part)
    
    # Send via Gmail SMTP
    with smtplib.SMTP_SSL('smtp.gmail.com', 465) as server:
        server.login(GMAIL_SENDER_EMAIL, GMAIL_APP_PASSWORD)
        server.sendmail(GMAIL_SENDER_EMAIL, to_email, msg.as_string())


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/transcribe', methods=['POST'])
def transcribe():
    try:
        logger.info("Received transcription request")
        
        if 'audio' not in request.files:
            logger.error("No audio file in request")
            return jsonify({'error': 'No audio file provided'}), 400
        
        file = request.files['audio']
        logger.info(f"File received: {file.filename}")
        
        if file.filename == '':
            logger.error("Empty filename")
            return jsonify({'error': 'No file selected'}), 400
        
        if not allowed_file(file.filename):
            logger.error(f"Invalid file format: {file.filename}")
            return jsonify({'error': 'Invalid file format. Supported formats: mp3, mp4, mpeg, mpga, m4a, wav, webm'}), 400
        
        filename = secure_filename(file.filename)
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(filename)[1]) as temp_file:
            file.save(temp_file.name)
            temp_file_path = temp_file.name
        
        logger.info(f"File saved temporarily to: {temp_file_path}")
        file_size = os.path.getsize(temp_file_path)
        logger.info(f"File size: {file_size} bytes")
        
        try:
            logger.info("Step 1: Transcribing with diarization model...")
            raw_transcript = transcribe_with_diarization(temp_file_path)
            logger.info(f"Raw transcript length: {len(raw_transcript)} chars")
            
            logger.info("Step 2: Identifying speakers with GPT-4...")
            diarized_result = identify_speakers_with_gpt4(raw_transcript)
            
            result = {
                'text': raw_transcript,
                'segments': diarized_result.get('segments', [])
            }
            
            logger.info(f"Returning {len(result['segments'])} diarized segments")
            return jsonify(result)
        
        finally:
            if os.path.exists(temp_file_path):
                os.unlink(temp_file_path)
                logger.info("Temporary file cleaned up")
    
    except Exception as e:
        logger.exception(f"Error during transcription: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/export/pdf', methods=['POST'])
def export_pdf():
    """Export transcript as PDF"""
    try:
        data = request.get_json()
        segments = data.get('segments', [])
        title = data.get('title', 'Meeting Transcript')
        
        pdf_content = generate_pdf(segments, title)
        
        return Response(
            pdf_content,
            mimetype='application/pdf',
            headers={'Content-Disposition': f'attachment; filename="{title}.pdf"'}
        )
    except Exception as e:
        logger.exception(f"Error generating PDF: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/export/text', methods=['POST'])
def export_text():
    """Export transcript as plain text"""
    try:
        data = request.get_json()
        segments = data.get('segments', [])
        title = data.get('title', 'Meeting Transcript')
        
        text_content = generate_text(segments)
        
        return Response(
            text_content,
            mimetype='text/plain',
            headers={'Content-Disposition': f'attachment; filename="{title}.txt"'}
        )
    except Exception as e:
        logger.exception(f"Error generating text: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/export/markdown', methods=['POST'])
def export_markdown():
    """Export transcript as Markdown"""
    try:
        data = request.get_json()
        segments = data.get('segments', [])
        title = data.get('title', 'Meeting Transcript')
        
        md_content = generate_markdown(segments, title)
        
        return Response(
            md_content,
            mimetype='text/markdown',
            headers={'Content-Disposition': f'attachment; filename="{title}.md"'}
        )
    except Exception as e:
        logger.exception(f"Error generating markdown: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/export/html', methods=['POST'])
def export_html():
    """Export transcript as HTML"""
    try:
        data = request.get_json()
        segments = data.get('segments', [])
        title = data.get('title', 'Meeting Transcript')
        
        html_content = generate_html(segments, title)
        
        return Response(
            html_content,
            mimetype='text/html',
            headers={'Content-Disposition': f'attachment; filename="{title}.html"'}
        )
    except Exception as e:
        logger.exception(f"Error generating HTML: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/send-email', methods=['POST'])
def send_transcript_email():
    """Send transcript via email"""
    try:
        data = request.get_json()
        to_email = data.get('email')
        segments = data.get('segments', [])
        title = data.get('title', 'Meeting Transcript')
        include_pdf = data.get('include_pdf', True)
        
        if not to_email:
            return jsonify({'error': 'Email address is required'}), 400
        
        # Generate text and HTML versions
        text_content = generate_text(segments)
        html_content = generate_html(segments, title)
        
        # Generate PDF attachment if requested
        pdf_attachment = None
        pdf_name = None
        if include_pdf:
            pdf_attachment = generate_pdf(segments, title)
            pdf_name = f"{title}.pdf"
        
        # Send email
        send_email(
            to_email=to_email,
            subject=f"Transcript: {title}",
            body_text=text_content,
            body_html=html_content,
            attachment=pdf_attachment,
            attachment_name=pdf_name
        )
        
        return jsonify({'success': True, 'message': f'Transcript sent to {to_email}'})
    
    except Exception as e:
        logger.exception(f"Error sending email: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/health')
def health():
    return jsonify({'status': 'healthy'})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Meeting Transcriber - AI-Powered Diarization')
    parser.add_argument('--port', type=int, default=None,
                        help='Port to run the server on (default: 5001, or PORT env variable)')
    args = parser.parse_args()
    
    if args.port:
        port = args.port
    else:
        port = int(os.environ.get('PORT', 5001))
    
    app.run(host='0.0.0.0', port=port, debug=False)
