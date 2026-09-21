import os
import re
import math
import html
import sqlite3
import difflib
from datetime import datetime
from flask import (
    Flask, render_template, request, redirect, url_for, flash, jsonify, g
)
from werkzeug.exceptions import RequestEntityTooLarge

# ---------------------------------------------------------------------------
# App Configuration
# ---------------------------------------------------------------------------
app = Flask(__name__)
app.config['SECRET_KEY'] = 'plagiarism_detector_secret_key_btech_cse_2026'
app.config['DATABASE'] = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'database.db')
app.config['UPLOAD_FOLDER'] = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static', 'uploads')
app.config['MAX_CONTENT_LENGTH'] = 2 * 1024 * 1024  # Maximum 2MB per upload

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# ---------------------------------------------------------------------------
# Check for Scikit-Learn (Graceful Fallback to Pure Python NLP)
# ---------------------------------------------------------------------------
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

# ---------------------------------------------------------------------------
# Stopwords List (Standard English Stopwords for Academic NLP)
# ---------------------------------------------------------------------------
STOP_WORDS = {
    'a', 'about', 'above', 'after', 'again', 'against', 'all', 'am', 'an', 'and',
    'any', 'are', 'aren\'t', 'as', 'at', 'be', 'because', 'been', 'before', 'being',
    'below', 'between', 'both', 'but', 'by', 'can\'t', 'cannot', 'could', 'couldn\'t',
    'did', 'didn\'t', 'do', 'does', 'doesn\'t', 'doing', 'don\'t', 'down', 'during',
    'each', 'few', 'for', 'from', 'further', 'had', 'hadn\'t', 'has', 'hasn\'t',
    'have', 'haven\'t', 'having', 'he', 'he\'d', 'he\'ll', 'he\'s', 'her', 'here',
    'here\'s', 'hers', 'herself', 'him', 'himself', 'his', 'how', 'how\'s', 'i',
    'i\'d', 'i\'ll', 'i\'m', 'i\'ve', 'if', 'in', 'into', 'is', 'isn\'t', 'it',
    'it\'s', 'its', 'itself', 'let\'s', 'me', 'more', 'most', 'mustn\'t', 'my',
    'myself', 'no', 'nor', 'not', 'of', 'off', 'on', 'once', 'only', 'or', 'other',
    'ought', 'our', 'ours', 'ourselves', 'out', 'over', 'own', 'same', 'shan\'t',
    'she', 'she\'d', 'she\'ll', 'she\'s', 'should', 'shouldn\'t', 'so', 'some',
    'such', 'than', 'that', 'that\'s', 'the', 'their', 'theirs', 'them', 'themselves',
    'then', 'there', 'there\'s', 'these', 'they', 'they\'d', 'they\'ll', 'they\'re',
    'they\'ve', 'this', 'those', 'through', 'to', 'too', 'under', 'until', 'up',
    'very', 'was', 'wasn\'t', 'we', 'we\'d', 'we\'ll', 'we\'re', 'we\'ve', 'were',
    'weren\'t', 'what', 'what\'s', 'when', 'when\'s', 'where', 'where\'s', 'which',
    'while', 'who', 'who\'s', 'whom', 'why', 'why\'s', 'with', 'won\'t', 'would',
    'wouldn\'t', 'you', 'you\'d', 'you\'ll', 'you\'re', 'you\'ve', 'your', 'yours',
    'yourself', 'yourselves'
}

# ---------------------------------------------------------------------------
# Database Management
# ---------------------------------------------------------------------------
def get_db():
    """Returns a SQLite database connection for the current application context."""
    db = getattr(g, '_database', None)
    if db is None:
        db = g._database = sqlite3.connect(app.config['DATABASE'])
        db.row_factory = sqlite3.Row
    return db

@app.teardown_appcontext
def close_connection(exception):
    """Closes the database connection at the end of each request."""
    db = getattr(g, '_database', None)
    if db is not None:
        db.close()

def init_db():
    """Initializes the SQLite database with the plagiarism_checks schema."""
    conn = sqlite3.connect(app.config['DATABASE'])
    try:
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS plagiarism_checks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                document1_word_count INTEGER NOT NULL,
                document2_word_count INTEGER NOT NULL,
                similarity_percentage REAL NOT NULL,
                difference_percentage REAL NOT NULL,
                matching_words INTEGER NOT NULL,
                unique_words INTEGER NOT NULL,
                status TEXT NOT NULL,
                duplicate_category TEXT NOT NULL,
                document1_text TEXT NOT NULL,
                document2_text TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
        ''')
        conn.commit()
    finally:
        conn.close()

# Ensure database is automatically initialized upon module import
init_db()

# ---------------------------------------------------------------------------
# Text Preprocessing & NLP Engine
# ---------------------------------------------------------------------------
WORD_REGEX = re.compile(r"\b[a-zA-Z0-9]+(?:'[a-zA-Z0-9]+)?\b")

def count_words(text):
    """Consistent word count across all routes and comparisons."""
    if not text:
        return 0
    return len(WORD_REGEX.findall(text))

def clean_word(token):
    """Normalize a word token by stripping punctuation and lowercasing."""
    return re.sub(r'[^a-z0-9]', '', token.lower())

def preprocess_text(text):
    """
    Cleans and normalizes text:
    - Tokenizes into words preserving contractions
    - Normalizes to lowercase and strips symbols
    - Filters out stop words (falls back if all were stopwords)
    Returns: (raw_words, filtered_tokens)
    """
    if not text:
        return [], []
    
    raw_words = WORD_REGEX.findall(text)
    cleaned_tokens = [clean_word(w) for w in raw_words]
    cleaned_tokens = [w for w in cleaned_tokens if w]
    
    filtered_tokens = [w for w in cleaned_tokens if w not in STOP_WORDS]
    if not filtered_tokens and cleaned_tokens:
        filtered_tokens = cleaned_tokens
        
    return raw_words, filtered_tokens

def pure_python_tfidf_cosine(tokens1, tokens2):
    """
    Pure Python implementation of TF-IDF Vectorization and Cosine Similarity:
    1. Builds joint vocabulary from tokens1 and tokens2.
    2. Calculates Term Frequency (TF): tf(t, d) = count(t, d) / total_tokens_d
    3. Smooth Inverse Document Frequency: idf(t) = ln((1 + N) / (1 + df(t))) + 1
    4. Computes TF-IDF weight vector for each document.
    5. Calculates Cosine Similarity = dot_product(v1, v2) / (norm(v1) * norm(v2))
    """
    if not tokens1 or not tokens2:
        return 0.0

    vocab = sorted(list(set(tokens1 + tokens2)))
    if not vocab:
        return 0.0

    df = {}
    for term in vocab:
        count = 0
        if term in tokens1:
            count += 1
        if term in tokens2:
            count += 1
        df[term] = count

    N = 2
    idf = {term: math.log((1 + N) / (1 + df[term])) + 1.0 for term in vocab}

    def get_tfidf_vector(tokens):
        total_tokens = len(tokens)
        counts = {}
        for t in tokens:
            counts[t] = counts.get(t, 0) + 1
        
        vec = []
        for term in vocab:
            tf = counts.get(term, 0) / total_tokens
            tfidf_val = tf * idf[term]
            vec.append(tfidf_val)
            
        norm = math.sqrt(sum(x * x for x in vec))
        if norm == 0:
            return [0.0] * len(vec)
        return [x / norm for x in vec]

    v1 = get_tfidf_vector(tokens1)
    v2 = get_tfidf_vector(tokens2)

    cosine_sim = sum(a * b for a, b in zip(v1, v2))
    return max(0.0, min(1.0, cosine_sim))

def calculate_similarity(text1, text2):
    """
    Calculates the similarity percentage between two documents using TF-IDF and Cosine Similarity.
    Uses scikit-learn when available, with a robust mathematical fallback.
    """
    raw1, tokens1 = preprocess_text(text1)
    raw2, tokens2 = preprocess_text(text2)

    if not tokens1 or not tokens2:
        return 0.0

    # Normalized exact sequence match check
    if tokens1 == tokens2:
        return 100.0

    clean_text1 = " ".join(tokens1)
    clean_text2 = " ".join(tokens2)

    if HAS_SKLEARN:
        try:
            vectorizer = TfidfVectorizer(token_pattern=r'(?u)\b\w+\b')
            tfidf_matrix = vectorizer.fit_transform([clean_text1, clean_text2])
            sim_matrix = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])
            score = float(sim_matrix[0][0]) * 100.0
        except Exception:
            score = pure_python_tfidf_cosine(tokens1, tokens2) * 100.0
    else:
        score = pure_python_tfidf_cosine(tokens1, tokens2) * 100.0

    return round(max(0.0, min(100.0, score)), 2)

def determine_status_and_category(similarity):
    """Maps the similarity percentage to user-facing status and duplicate detection category."""
    if similarity >= 81.0:
        status = "Very High Similarity"
    elif similarity >= 61.0:
        status = "High Similarity"
    elif similarity >= 41.0:
        status = "Moderate Similarity"
    elif similarity >= 21.0:
        status = "Low Similarity"
    else:
        status = "Very Low Similarity"

    if similarity >= 98.0:
        category = "Exact Duplicate"
    elif similarity >= 75.0:
        category = "Highly Similar"
    elif similarity >= 35.0:
        category = "Partially Similar"
    else:
        category = "Mostly Different"

    return status, category

# ---------------------------------------------------------------------------
# Multi-Pass Side-by-Side Highlight Generator
# ---------------------------------------------------------------------------
def generate_highlighted_comparison(text1, text2):
    """
    Generates synchronized HTML side-by-side comparison with multi-pass matching:
    1. Tokenizes into words, spaces, and punctuation preserving complete layout.
    2. Runs multi-pass phrase extraction to detect verbatim blocks even if sentences/paragraphs are transposed.
    3. Matches remaining isolated common words.
    4. Calculates true word counts and unmatched differences.
    """
    if not text1:
        text1 = ""
    if not text2:
        text2 = ""

    # Token pattern: words with apostrophes, spaces, or individual punctuation
    token_pattern = re.compile(r"[a-zA-Z0-9]+(?:'[a-zA-Z0-9]+)?|\s+|[^\w\s]")
    tokens1 = token_pattern.findall(text1) if text1 else []
    tokens2 = token_pattern.findall(text2) if text2 else []

    # Identify word tokens and their token indices
    words1_info = [(i, clean_word(t)) for i, t in enumerate(tokens1) if clean_word(t)]
    words2_info = [(i, clean_word(t)) for i, t in enumerate(tokens2) if clean_word(t)]

    w1_list = [w for _, w in words1_info]
    w2_list = [w for _, w in words2_info]

    matched_w1_indices = set()
    matched_w2_indices = set()

    # Pass 1: Multi-pass sequence matching on words (longest common phrases first, >= 2 words)
    # This detects transposed paragraphs and reordered sentences
    min_phrase_len = 2
    active_len = min(len(w1_list), len(w2_list))

    while active_len >= min_phrase_len:
        best_match = None
        best_size = 0
        best_a = -1
        best_b = -1

        matcher = difflib.SequenceMatcher(None, w1_list, w2_list, autojunk=False)
        for block in matcher.get_matching_blocks():
            if block.size >= min_phrase_len:
                # Check if any word in this block is already claimed
                overlap = False
                for k in range(block.size):
                    if (block.a + k) in matched_w1_indices or (block.b + k) in matched_w2_indices:
                        overlap = True
                        break
                if not overlap and block.size > best_size:
                    best_size = block.size
                    best_a = block.a
                    best_b = block.b

        if best_size >= min_phrase_len:
            for k in range(best_size):
                matched_w1_indices.add(best_a + k)
                matched_w2_indices.add(best_b + k)
        else:
            break

    # Pass 2: Match remaining single common words if available
    available_w2 = {}
    for idx2, w in enumerate(w2_list):
        if idx2 not in matched_w2_indices:
            available_w2.setdefault(w, []).append(idx2)

    for idx1, w in enumerate(w1_list):
        if idx1 not in matched_w1_indices and w in available_w2 and available_w2[w]:
            matched_idx2 = available_w2[w].pop(0)
            matched_w1_indices.add(idx1)
            matched_w2_indices.add(matched_idx2)

    # Map matched word indices back to token indices
    token_match_set1 = {words1_info[idx][0] for idx in matched_w1_indices}
    token_match_set2 = {words2_info[idx][0] for idx in matched_w2_indices}

    # Render Document 1
    html1_parts = []
    for idx, token in enumerate(tokens1):
        escaped_token = html.escape(token)
        is_word = bool(clean_word(token))
        if idx in token_match_set1:
            html1_parts.append(f'<mark class="text-match" title="Matching Word / Phrase">{escaped_token}</mark>')
        elif is_word:
            html1_parts.append(f'<span class="text-unique-1" title="Unique to Document 1">{escaped_token}</span>')
        else:
            html1_parts.append(escaped_token)

    # Render Document 2
    html2_parts = []
    for idx, token in enumerate(tokens2):
        escaped_token = html.escape(token)
        is_word = bool(clean_word(token))
        if idx in token_match_set2:
            html2_parts.append(f'<mark class="text-match" title="Matching Word / Phrase">{escaped_token}</mark>')
        elif is_word:
            html2_parts.append(f'<span class="text-unique-2" title="Unique / Different in Document 2">{escaped_token}</span>')
        else:
            html2_parts.append(escaped_token)

    # Accurate word stats
    doc1_words = len(w1_list)
    doc2_words = len(w2_list)
    matching_words_count = len(matched_w1_indices)
    unique_words_count = (doc1_words - matching_words_count) + (doc2_words - len(matched_w2_indices))

    return (
        "".join(html1_parts),
        "".join(html2_parts),
        matching_words_count,
        unique_words_count
    )

# ---------------------------------------------------------------------------
# Routes & Controllers
# ---------------------------------------------------------------------------

@app.route('/')
def home():
    """Renders the Home Page with hero banner, live stats, and how it works."""
    db = get_db()
    row = db.execute('SELECT COUNT(*) as count FROM plagiarism_checks').fetchone()
    total_checks = row['count'] if row else 0
    return render_template('index.html', total_checks=total_checks)

@app.route('/checker', methods=['GET', 'POST'])
def checker():
    """
    Plagiarism Check Page:
    - GET: Renders input textareas & upload zones.
    - POST: Validates documents, calculates similarity, saves to SQLite, redirects to result.
    """
    if request.method == 'POST':
        doc1_text = request.form.get('doc1_text', '').strip()
        doc2_text = request.form.get('doc2_text', '').strip()

        # Handle Document 1 file upload
        if 'doc1_file' in request.files:
            file1 = request.files['doc1_file']
            if file1 and file1.filename:
                allowed_exts = ('.txt', '.md', '.py', '.c', '.cpp', '.java', '.js', '.html', '.css', '.json')
                if not file1.filename.lower().endswith(allowed_exts):
                    flash(f'Document 1 format not supported. Allowed extensions: {", ".join(allowed_exts)}', 'danger')
                    return render_template('checker.html', doc1_text=doc1_text, doc2_text=doc2_text)
                try:
                    raw_bytes = file1.read()
                    file1_content = raw_bytes.decode('utf-8-sig', errors='replace').strip()
                    if file1_content:
                        doc1_text = file1_content
                except Exception as e:
                    flash(f'Error reading Document 1 file: {str(e)}', 'danger')
                    return render_template('checker.html', doc1_text=doc1_text, doc2_text=doc2_text)

        # Handle Document 2 file upload
        if 'doc2_file' in request.files:
            file2 = request.files['doc2_file']
            if file2 and file2.filename:
                allowed_exts = ('.txt', '.md', '.py', '.c', '.cpp', '.java', '.js', '.html', '.css', '.json')
                if not file2.filename.lower().endswith(allowed_exts):
                    flash(f'Document 2 format not supported. Allowed extensions: {", ".join(allowed_exts)}', 'danger')
                    return render_template('checker.html', doc1_text=doc1_text, doc2_text=doc2_text)
                try:
                    raw_bytes = file2.read()
                    file2_content = raw_bytes.decode('utf-8-sig', errors='replace').strip()
                    if file2_content:
                        doc2_text = file2_content
                except Exception as e:
                    flash(f'Error reading Document 2 file: {str(e)}', 'danger')
                    return render_template('checker.html', doc1_text=doc1_text, doc2_text=doc2_text)

        # Validation Checks
        if not doc1_text and not doc2_text:
            flash('Please provide text or upload a document file for both documents.', 'warning')
            return render_template('checker.html', doc1_text=doc1_text, doc2_text=doc2_text)

        if not doc1_text:
            flash('Original Document 1 is empty. Please enter or upload text.', 'warning')
            return render_template('checker.html', doc1_text=doc1_text, doc2_text=doc2_text)

        if not doc2_text:
            flash('Document 2 (Suspected Copy) is empty. Please enter or upload text.', 'warning')
            return render_template('checker.html', doc1_text=doc1_text, doc2_text=doc2_text)

        words1 = count_words(doc1_text)
        words2 = count_words(doc2_text)

        if words1 < 3:
            flash('Document 1 is too short. Please provide at least 3 words for meaningful analysis.', 'warning')
            return render_template('checker.html', doc1_text=doc1_text, doc2_text=doc2_text)

        if words2 < 3:
            flash('Document 2 is too short. Please provide at least 3 words for meaningful analysis.', 'warning')
            return render_template('checker.html', doc1_text=doc1_text, doc2_text=doc2_text)

        # Run Similarity Analysis
        similarity = calculate_similarity(doc1_text, doc2_text)
        difference = round(100.0 - similarity, 2)
        status, category = determine_status_and_category(similarity)

        # Side-by-side matching
        _, _, matching_words, unique_words = generate_highlighted_comparison(doc1_text, doc2_text)

        # Save record in SQLite database
        now_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        db = get_db()
        cursor = db.cursor()
        cursor.execute('''
            INSERT INTO plagiarism_checks (
                document1_word_count,
                document2_word_count,
                similarity_percentage,
                difference_percentage,
                matching_words,
                unique_words,
                status,
                duplicate_category,
                document1_text,
                document2_text,
                created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            words1,
            words2,
            similarity,
            difference,
            matching_words,
            unique_words,
            status,
            category,
            doc1_text,
            doc2_text,
            now_str
        ))
        db.commit()
        new_check_id = cursor.lastrowid

        flash('Plagiarism analysis completed successfully!', 'success')
        return redirect(url_for('result', check_id=new_check_id))

    return render_template('checker.html')

@app.route('/result/<int:check_id>')
def result(check_id):
    """
    Renders the Similarity Result Page:
    - Circular percentage meters
    - Academic duplicate verdict badge
    - Synchronized side-by-side highlighted comparison
    """
    db = get_db()
    record = db.execute('SELECT * FROM plagiarism_checks WHERE id = ?', (check_id,)).fetchone()

    if not record:
        flash(f'Analysis record #{check_id} was not found.', 'danger')
        return redirect(url_for('checker'))

    highlighted_doc1, highlighted_doc2, matching_words, unique_words = generate_highlighted_comparison(
        record['document1_text'],
        record['document2_text']
    )

    return render_template(
        'result.html',
        record=record,
        highlighted_doc1=highlighted_doc1,
        highlighted_doc2=highlighted_doc2,
        matching_words=matching_words,
        unique_words=unique_words,
        engine_type="Scikit-Learn TF-IDF" if HAS_SKLEARN else "Pure Python NLP Engine (Built-in)"
    )

@app.route('/history')
def history():
    """Renders the Check History Page with previous checks and action buttons."""
    db = get_db()
    checks = db.execute('''
        SELECT id, document1_word_count, document2_word_count,
               similarity_percentage, difference_percentage, matching_words, unique_words,
               status, duplicate_category, created_at
        FROM plagiarism_checks
        ORDER BY id DESC
    ''').fetchall()
    return render_template('history.html', checks=checks)

@app.route('/history/delete/<int:check_id>', methods=['POST'])
def delete_history(check_id):
    """Deletes a specific check record from SQLite database."""
    db = get_db()
    cursor = db.cursor()
    cursor.execute('DELETE FROM plagiarism_checks WHERE id = ?', (check_id,))
    db.commit()
    if cursor.rowcount > 0:
        flash(f'Check #{check_id} deleted successfully.', 'info')
    else:
        flash(f'Check #{check_id} could not be found.', 'warning')
    return redirect(url_for('history'))

@app.route('/history/clear', methods=['POST'])
def clear_history():
    """Clears all previous checks from SQLite database."""
    db = get_db()
    db.execute('DELETE FROM plagiarism_checks')
    db.commit()
    flash('All check history has been cleared successfully.', 'info')
    return redirect(url_for('history'))

@app.route('/dashboard')
def dashboard():
    """Renders the Analytics Dashboard with statistical cards and recent activity."""
    db = get_db()
    row = db.execute('SELECT COUNT(*) as count FROM plagiarism_checks').fetchone()
    total_checks = row['count'] if row else 0

    if total_checks > 0:
        stats = db.execute('''
            SELECT 
                AVG(similarity_percentage) as avg_sim,
                MAX(similarity_percentage) as max_sim,
                MIN(similarity_percentage) as min_sim
            FROM plagiarism_checks
        ''').fetchone()

        avg_similarity = round(stats['avg_sim'], 2) if stats['avg_sim'] is not None else 0.0
        max_similarity = round(stats['max_sim'], 2) if stats['max_sim'] is not None else 0.0
        min_similarity = round(stats['min_sim'], 2) if stats['min_sim'] is not None else 0.0

        status_counts = {
            'Very High Similarity': 0,
            'High Similarity': 0,
            'Moderate Similarity': 0,
            'Low Similarity': 0,
            'Very Low Similarity': 0,
        }
        for r in db.execute("SELECT status, COUNT(*) as c FROM plagiarism_checks GROUP BY status").fetchall():
            if r['status'] in status_counts:
                status_counts[r['status']] = r['c']

        category_counts = {
            'Exact Duplicate': 0,
            'Highly Similar': 0,
            'Partially Similar': 0,
            'Mostly Different': 0,
        }
        for r in db.execute("SELECT duplicate_category, COUNT(*) as c FROM plagiarism_checks GROUP BY duplicate_category").fetchall():
            if r['duplicate_category'] in category_counts:
                category_counts[r['duplicate_category']] = r['c']

        recent_checks = db.execute('''
            SELECT id, document1_word_count, document2_word_count,
                   similarity_percentage, status, duplicate_category, created_at
            FROM plagiarism_checks
            ORDER BY id DESC
            LIMIT 5
        ''').fetchall()
    else:
        avg_similarity = 0.0
        max_similarity = 0.0
        min_similarity = 0.0
        status_counts = {
            'Very High Similarity': 0,
            'High Similarity': 0,
            'Moderate Similarity': 0,
            'Low Similarity': 0,
            'Very Low Similarity': 0,
        }
        category_counts = {
            'Exact Duplicate': 0,
            'Highly Similar': 0,
            'Partially Similar': 0,
            'Mostly Different': 0,
        }
        recent_checks = []

    return render_template(
        'dashboard.html',
        total_checks=total_checks,
        avg_similarity=avg_similarity,
        max_similarity=max_similarity,
        min_similarity=min_similarity,
        status_counts=status_counts,
        category_counts=category_counts,
        recent_checks=recent_checks
    )

@app.route('/api/check', methods=['POST'])
def api_check():
    """JSON API endpoint for programmatic or asynchronous similarity checks."""
    data = request.get_json(silent=True) or request.form
    doc1 = data.get('doc1', '').strip() if data else ''
    doc2 = data.get('doc2', '').strip() if data else ''

    if not doc1 or not doc2:
        return jsonify({'error': 'Both doc1 and doc2 are required.'}), 400

    words1 = count_words(doc1)
    words2 = count_words(doc2)

    if words1 < 3 or words2 < 3:
        return jsonify({
            'error': 'Both documents must contain at least 3 words for analysis.',
            'doc1_words': words1,
            'doc2_words': words2
        }), 400

    similarity = calculate_similarity(doc1, doc2)
    difference = round(100.0 - similarity, 2)
    status, category = determine_status_and_category(similarity)
    _, _, matching_words, unique_words = generate_highlighted_comparison(doc1, doc2)

    return jsonify({
        'similarity_percentage': similarity,
        'difference_percentage': difference,
        'status': status,
        'duplicate_category': category,
        'doc1_words': words1,
        'doc2_words': words2,
        'matching_words': matching_words,
        'unique_words': unique_words,
        'disclaimer': 'Similarity percentage indicates textual similarity only. A high similarity score does not automatically prove plagiarism.'
    })

# ---------------------------------------------------------------------------
# Error Handlers
# ---------------------------------------------------------------------------
@app.errorhandler(413)
def request_entity_too_large(error):
    """Handles file uploads exceeding MAX_CONTENT_LENGTH."""
    if request.is_json or request.path.startswith('/api/'):
        return jsonify({'error': 'Uploaded file exceeds the maximum allowed limit of 2MB.'}), 413
    flash('File upload failed: Uploaded file exceeds the 2MB size limit.', 'danger')
    return redirect(url_for('checker'))

@app.errorhandler(404)
def page_not_found(error):
    if request.is_json or request.path.startswith('/api/'):
        return jsonify({'error': 'Resource not found.'}), 404
    return render_template('errors/404.html'), 404

@app.errorhandler(500)
def internal_server_error(error):
    if request.is_json or request.path.startswith('/api/'):
        return jsonify({'error': 'Internal server error occurred.'}), 500
    return render_template('errors/500.html'), 500

# ---------------------------------------------------------------------------
# Application Entry Point
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    print("Database initialized successfully.")
    print(f"NLP Engine: {'Scikit-Learn TF-IDF' if HAS_SKLEARN else 'Pure-Python Mathematical TF-IDF (Built-in)'}")
    print("Starting Flask server at http://127.0.0.1:5000 ...")
    app.run(debug=True, host='127.0.0.1', port=5000)
