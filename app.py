import requests
import re
from flask import Flask, request, jsonify
from collections import Counter
import math
from flask_cors import CORS

# Konfigurasi API dan Aplikasi Flask
API_KEY = "2330fe54e9254616b5c33a4ff03c06fc"  # Masukkan API Key Anda
NEWS_API_URL = "https://newsapi.org/v2/everything"
app = Flask(__name__)

# Mengaktifkan CORS
CORS(app)

# Daftar stopwords sederhana
stopwords = set([
    "i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours", "yourself", "yourselves", 
    "he", "him", "his", "himself", "she", "her", "hers", "herself", "it", "its", "itself", "they", "them", "their", 
    "theirs", "themselves", "what", "which", "who", "whom", "this", "that", "these", "those", "am", "is", "are", "was", 
    "were", "be", "been", "being", "have", "has", "had", "having", "do", "does", "did", "doing", "a", "an", "the", "and", 
    "but", "if", "or", "because", "as", "until", "while", "of", "at", "by", "for", "with", "about", "against", "between", 
    "into", "through", "during", "before", "after", "above", "below", "to", "from", "up", "down", "in", "out", "on", "off", 
    "over", "under", "again", "further", "then", "once", "here", "there", "when", "where", "why", "how", "all", "any", 
    "both", "each", "few", "more", "most", "other", "some", "such", "no", "nor", "not", "only", "own", "same", "so", "than", 
    "too", "very", "s", "t", "can", "will", "just", "don", "should", "now", "d", "ll", "m", "o", "re", "ve", "y", "ain", "aren", 
    "couldn", "didn", "doesn", "hadn", "hasn", "haven", "isn", "ma", "mightn", "mustn", "needn", "shan", "shouldn", "wasn", 
    "weren", "won", "wouldn"
])

# Manual Tokenization
def tokenize(text):
    """
    Manual Tokenization (memecah teks menjadi token berdasarkan spasi dan tanda baca)
    """
    return re.findall(r'\b\w+\b', text.lower())  # Menggunakan regex untuk mengambil kata

# Manual Stemming (Sederhana)
def stem(word):
    """
    Stemming sederhana: Menghapus akhiran umum seperti "ing", "ly", "ed", "es", dan lainnya.
    """
    suffixes = ['ing', 'ly', 'ed', 'es', 's']
    for suffix in suffixes:
        if word.endswith(suffix):
            return word[:-len(suffix)]
    return word

# Preprocessing Tools
def preprocess_text(text):
    """
    Membersihkan teks (tokenisasi, stopwords removal, stemming, dan menghapus karakter khusus serta angka).
    """
    # Tokenisasi manual
    tokens = tokenize(text)
    
    # Hapus stopwords secara manual
    tokens = [word for word in tokens if word not in stopwords]
    
    # Stemming manual
    tokens = [stem(word) for word in tokens]
    
    return " ".join(tokens)  # Gabungkan kembali token menjadi sebuah string

# Menghitung TF (Term Frequency) manual
def compute_tf(doc):
    tf = Counter(doc)
    total_terms = len(doc)
    for term in tf:
        tf[term] = tf[term] / total_terms  # Normalisasi
    return tf

# Menghitung IDF (Inverse Document Frequency) manual
def compute_idf(corpus):
    N = len(corpus)
    df = {}
    for doc in corpus:
        tokens = set(doc)  # Set untuk menghitung term yang unik dalam dokumen
        for token in tokens:
            df[token] = df.get(token, 0) + 1
    idf = {}
    for term, doc_count in df.items():
        idf[term] = math.log((N - doc_count + 0.5) / (doc_count + 0.5) + 1.0)
    return idf

# Menghitung TF-IDF
def compute_tfidf(corpus):
    tfidf = []
    idf = compute_idf(corpus)
    for doc in corpus:
        tf = compute_tf(doc)
        tfidf_doc = {}
        for term, tf_val in tf.items():
            tfidf_doc[term] = tf_val * idf.get(term, 0)
        tfidf.append(tfidf_doc)
    return tfidf

# Menghitung BM25
def compute_bm25(corpus, k1=1.5, b=0.75):
    N = len(corpus)
    avgdl = sum(len(doc) for doc in corpus) / N
    idf = compute_idf(corpus)
    bm25_scores = []

    for doc in corpus:
        tf = compute_tf(doc)
        bm25_doc = {}
        for term, tf_val in tf.items():
            # Perhitungan BM25 untuk setiap term dalam dokumen
            idf_val = idf.get(term, 0)
            tf_term = tf_val
            doc_len = len(doc)
            bm25_score = idf_val * ((tf_term * (k1 + 1)) / (tf_term + k1 * (1 - b + b * doc_len / avgdl)))
            bm25_doc[term] = bm25_score
        bm25_scores.append(bm25_doc)

    return bm25_scores

# Ambil Data dari NewsAPI
def fetch_articles(query):
    """
    Mengambil data artikel dari NewsAPI berdasarkan kata kunci (query).
    """
    params = {
        'q': query,
        'language': 'en',  # Menggunakan bahasa Inggris
        'pageSize': 100,   # Maksimal 100 artikel
        'apiKey': API_KEY
    }
    response = requests.get(NEWS_API_URL, params=params)
    if response.status_code == 200:
        data = response.json()
        articles = data.get('articles', [])
        print(f"Fetched {len(articles)} articles for query '{query}'")  # Debug log
        return articles
    else:
        print(f"Failed to fetch articles for query '{query}'. Status code: {response.status_code}")  # Debug log
        return []

# Inisialisasi Data Artikel
articles = []
cleaned_texts = []
vectorizer = None
bm25 = None
tfidf_matrix = None

@app.route('/search', methods=['GET'])
def search_articles():
    """
    Endpoint untuk mencari artikel berdasarkan query menggunakan TF-IDF dan BM25.
    """
    global articles, cleaned_texts, bm25, vectorizer, tfidf_matrix

    query = request.args.get('query', '')
    if not query:
        return jsonify({"error": "Query parameter is required."}), 400

    # Ambil artikel dari NewsAPI berdasarkan query
    articles = fetch_articles(query)

    if len(articles) == 0:
        return jsonify({"error": "No articles found for the given query."}), 400

    # Gabungkan judul, deskripsi, dan konten
    texts = [
        f"{article['title']} {article['description']} {article.get('content', '')}"
        for article in articles
    ]

    # Preprocessing teks
    cleaned_texts = [preprocess_text(text) for text in texts]

    # Tokenisasi untuk perhitungan TF-IDF dan BM25
    tokenized_corpus = [text.split() for text in cleaned_texts]

    # Hitung TF-IDF secara manual
    tfidf_scores = compute_tfidf(tokenized_corpus)

    # Hitung BM25 secara manual
    bm25_scores = compute_bm25(tokenized_corpus)

    # Pencarian berdasarkan skor (gabungan TF-IDF dan BM25)
    processed_query = preprocess_text(query)  # Proses query pengguna
    query_tokens = processed_query.split()

    # Hitung skor gabungan TF-IDF dan BM25 untuk setiap dokumen
    combined_scores = []
    for idx in range(len(tokenized_corpus)):
        tfidf_score = sum(tfidf_scores[idx].get(token, 0) for token in query_tokens)
        bm25_score = sum(bm25_scores[idx].get(token, 0) for token in query_tokens)
        combined_score = 0.7 * tfidf_score + 0.3 * bm25_score
        combined_scores.append((idx, combined_score))

    # Urutkan hasil berdasarkan skor gabungan
    combined_scores = sorted(combined_scores, key=lambda x: x[1], reverse=True)

    # Ambil top artikel
    top_results = []
    for idx, score in combined_scores[:50]:
        tfidf_score = sum(tfidf_scores[idx].get(token, 0) for token in query_tokens)
        bm25_score = sum(bm25_scores[idx].get(token, 0) for token in query_tokens)
        top_results.append({
            "title": articles[idx]['title'],
            "description": articles[idx]['description'],
            "url": articles[idx]['url'],
            "tfidf_score": tfidf_score,  # TF-IDF Score untuk dokumen ini
            "bm25_score": bm25_score,    # BM25 Score untuk dokumen ini
            "combined_score": score      # Skor gabungan untuk dokumen ini
        })


    return jsonify({"query": query, "results": top_results})

if __name__ == '__main__':
    app.run(debug=True)
