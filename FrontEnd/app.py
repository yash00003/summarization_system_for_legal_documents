from flask import Flask, render_template, request, send_file, jsonify
import os
import fitz  # PyMuPDF
from docx import Document
import subprocess
import tempfile
import io
from reportlab.platypus import SimpleDocTemplate, Paragraph
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
import sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from model.extractive.mmr import generate_mmr_summary
from model.extractive.caseSummarizer import generate_summary
from model.abstractive.LED.main import generate_led_summary
from model.abstractive.pegasus.main import pegasus_summarize
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {'pdf', 'doc', 'docx','txt'}


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def convert_to_text(file_path, filename):
    text = ""
    ext = filename.split('.')[-1].lower()
    
    try:
        if ext == 'pdf':
            doc = fitz.open(file_path)
            for page in doc:
                text += page.get_text()
        elif ext == 'docx':
            doc = Document(file_path)
            text = '\n'.join([para.text for para in doc.paragraphs if para.text.strip()])
        elif ext == 'doc':
            result = subprocess.run(
                ['antiword', file_path],
                capture_output=True,
                text=True,
                check=True
            )
            text = result.stdout.strip()
        elif ext == 'txt':  # Add this new case
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
    except Exception as e:
        raise RuntimeError(f"Conversion error: {str(e)}")
    
    return text

def generate_summaries(text):
    # Replace this with actual summarization logic
    # This is just placeholder content
    return {
        "brief": generate_mmr_summary(text),
        "detailed": generate_summary(text),
        "key_points": generate_led_summary(text),
        "technical": pegasus_summarize(text)
    }

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
            
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400

        if file and allowed_file(file.filename):
            try:
                filename = file.filename
                temp_path = os.path.join(tempfile.gettempdir(), filename)
                file.save(temp_path)
                
                text = convert_to_text(temp_path, filename)
                summaries = generate_summaries(text)
                
                os.remove(temp_path)
                return jsonify({
                    'summaries': summaries,
                    'status': 'success'
                })

            except Exception as e:
                return jsonify({'error': str(e)}), 500
        else:
            return jsonify({'error': 'Invalid file type'}), 400

    return render_template('index.html')

def text_to_pdf(text):
    buffer = io.BytesIO()
    
    doc = SimpleDocTemplate(buffer, 
                          pagesize=letter,
                          leftMargin=inch,
                          rightMargin=inch,
                          topMargin=inch,
                          bottomMargin=inch)
    
    styles = getSampleStyleSheet()
    style = styles["BodyText"]
    style.wordWrap = 'CJK'
    style.leading = 14
    style.fontSize = 12
    story = []
    paragraphs = text.split('\n')
    
    for para in paragraphs:
        clean_para = para.replace('\t', '&nbsp;'*4).strip() or '&nbsp;'
        p = Paragraph(clean_para, style)
        story.append(p)
    
    doc.build(story)
    buffer.seek(0)
    return buffer

@app.route('/save', methods=['POST'])
def save_text():
    try:
        data = request.json
        filename = data.get('filename', 'document').strip() or 'document'
        content = "\n\n".join([
            data.get('brief', ''),
            data.get('detailed', ''),
            data.get('key_points', ''),
            data.get('technical', '')
        ])
        
        pdf_buffer = text_to_pdf(content)
        
        return send_file(
            pdf_buffer,
            mimetype='application/pdf',
            as_attachment=True,
            download_name=f"{filename}.pdf"
        )
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    app.run(debug=True)