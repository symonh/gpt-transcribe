import os
import argparse
import logging
import requests
import json
import smtplib
import tempfile
import secrets
import redis
from functools import wraps
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.application import MIMEApplication
from flask import Flask, request, jsonify, render_template, Response, session, redirect, url_for
from openai import OpenAI
from werkzeug.utils import secure_filename
from werkzeug.security import check_password_hash, generate_password_hash
from io import BytesIO
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.enums import TA_LEFT
import markdown as md
from rq import Queue
from rq.job import Job

# Configuration - try to import config.py, fall back to environment variables
try:
    import config
    OPENAI_API_KEY = config.OPENAI_API_KEY
    GMAIL_SENDER_EMAIL = config.GMAIL_SENDER_EMAIL
    GMAIL_APP_PASSWORD = config.GMAIL_APP_PASSWORD
    MAX_CONTENT_LENGTH = config.MAX_CONTENT_LENGTH
    ALLOWED_EXTENSIONS = config.ALLOWED_EXTENSIONS
    # Auth credentials
    APP_USERNAME = getattr(config, 'APP_USERNAME', 'admin')
    APP_PASSWORD_HASH = getattr(config, 'APP_PASSWORD_HASH', None)
    SECRET_KEY = getattr(config, 'SECRET_KEY', secrets.token_hex(32))
    REDIS_URL = getattr(config, 'REDIS_URL', 'redis://localhost:6379')
except ImportError:
    # Fall back to environment variables (for Heroku deployment)
    OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')
    GMAIL_SENDER_EMAIL = os.environ.get('GMAIL_SENDER_EMAIL')
    GMAIL_APP_PASSWORD = os.environ.get('GMAIL_APP_PASSWORD')
    MAX_CONTENT_LENGTH = 300 * 1024 * 1024  # 300MB
    ALLOWED_EXTENSIONS = {'mp3', 'mp4', 'mpeg', 'mpga', 'm4a', 'wav', 'webm', 'qta', 'mov', 'aac', 'ogg', 'flac', 'wma'}
    # Auth credentials from environment
    APP_USERNAME = os.environ.get('APP_USERNAME', 'admin')
    APP_PASSWORD_HASH = os.environ.get('APP_PASSWORD_HASH')
    SECRET_KEY = os.environ.get('SECRET_KEY', secrets.token_hex(32))
    REDIS_URL = os.environ.get('REDIS_URL', 'redis://localhost:6379')

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH
app.config['SECRET_KEY'] = SECRET_KEY

# Initialize OpenAI client
client = OpenAI(api_key=OPENAI_API_KEY)

# Initialize Redis connection and queue
# Heroku Redis uses self-signed certs, need to disable SSL verification
try:
    if REDIS_URL.startswith('rediss://'):
        redis_conn = redis.from_url(REDIS_URL, ssl_cert_reqs=None)
    else:
        redis_conn = redis.from_url(REDIS_URL)
    task_queue = Queue('transcription', connection=redis_conn)
    REDIS_AVAILABLE = True
    logger.info("Redis connection established")
except Exception as e:
    logger.warning(f"Redis not available, falling back to sync mode: {e}")
    REDIS_AVAILABLE = False
    redis_conn = None
    task_queue = None


def login_required(f):
    """Decorator to require login for routes"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not session.get('logged_in'):
            if request.is_json:
                return jsonify({'error': 'Authentication required'}), 401
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


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


# ============ Authentication Routes ============

@app.route('/login', methods=['GET', 'POST'])
def login():
    """Handle login"""
    if session.get('logged_in'):
        return redirect(url_for('index'))
    
    error = None
    if request.method == 'POST':
        username = request.form.get('username', '')
        password = request.form.get('password', '')
        
        # Check credentials
        if username == APP_USERNAME and APP_PASSWORD_HASH:
            if check_password_hash(APP_PASSWORD_HASH, password):
                session['logged_in'] = True
                session['username'] = username
                session.permanent = True
                return redirect(url_for('index'))
        
        error = 'Invalid credentials'
    
    return render_template('login.html', error=error)


@app.route('/logout')
def logout():
    """Handle logout"""
    session.clear()
    return redirect(url_for('login'))


# ============ Protected Routes ============

@app.route('/')
@login_required
def index():
    return render_template('index.html')


@app.route('/transcribe', methods=['POST'])
@login_required
def transcribe():
    """Start a transcription job (async with Redis, sync fallback)"""
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
            return jsonify({'error': 'Invalid file format. Supported formats: mp3, mp4, mpeg, mpga, m4a, wav, webm, qta'}), 400
        
        filename = secure_filename(file.filename)
        
        # Read file content and encode as base64 to pass through Redis
        import base64
        file_content = file.read()
        file_data_b64 = base64.b64encode(file_content).decode('utf-8')
        
        logger.info(f"File size: {len(file_content)} bytes")
        
        # Queue the job if Redis is available
        if REDIS_AVAILABLE and task_queue:
            from jobs import transcribe_audio_job
            job = task_queue.enqueue(
                transcribe_audio_job,
                file_data_b64,
                filename,
                job_timeout=1800  # 30 minute timeout for large files with chunking
            )
            logger.info(f"Job queued with ID: {job.id}")
            return jsonify({
                'status': 'queued',
                'job_id': job.id,
                'message': 'Transcription started. Poll /job/<job_id> for status.'
            })
        else:
            # Fallback to sync processing (for local dev without Redis)
            logger.warning("Redis not available, processing synchronously")
            from jobs import transcribe_audio_job
            result = transcribe_audio_job(file_data_b64, filename)
            if result.get('status') == 'completed':
                return jsonify({
                    'status': 'completed',
                    'text': result.get('text', ''),
                    'segments': result.get('segments', []),
                    'duration': result.get('duration', 0)
                })
            else:
                return jsonify({'error': result.get('error', 'Unknown error')}), 500
    
    except Exception as e:
        logger.exception(f"Error during transcription: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/job/<job_id>')
@login_required
def get_job_status(job_id):
    """Check the status of a transcription job"""
    if not REDIS_AVAILABLE:
        return jsonify({'error': 'Redis not available'}), 503
    
    try:
        job = Job.fetch(job_id, connection=redis_conn)
        
        if job.is_finished:
            result = job.result
            if result.get('status') == 'completed':
                return jsonify({
                    'status': 'completed',
                    'text': result.get('text', ''),
                    'segments': result.get('segments', []),
                    'duration': result.get('duration', 0)
                })
            else:
                return jsonify({
                    'status': 'failed',
                    'error': result.get('error', 'Unknown error')
                })
        elif job.is_failed:
            return jsonify({
                'status': 'failed',
                'error': str(job.exc_info) if job.exc_info else 'Job failed'
            })
        elif job.is_started:
            return jsonify({'status': 'processing'})
        else:
            return jsonify({'status': 'queued'})
    
    except Exception as e:
        logger.exception(f"Error fetching job {job_id}: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/export/pdf', methods=['POST'])
@login_required
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
@login_required
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
@login_required
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
@login_required
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
@login_required
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
    return jsonify({
        'status': 'healthy',
        'redis': REDIS_AVAILABLE
    })


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
