# ==========================================
# TRUTHLENS - FAKE NEWS DETECTION SYSTEM
# ==========================================

from flask import Flask, render_template, request, redirect
import joblib
import sqlite3
from datetime import datetime
import trafilatura


# ==========================================
# FLASK APP
# ==========================================

app = Flask(__name__)


# ==========================================
# LOAD MACHINE LEARNING MODEL
# ==========================================

# V2 model trained using:
# Fake.csv + True.csv
# Title + Article Text

model = joblib.load("model/fake_news_model_v2.pkl")
vectorizer = joblib.load("model/vectorizer_v2.pkl")


# ==========================================
# DATABASE INITIALIZATION
# ==========================================

def init_db():

    conn = sqlite3.connect("history.db")

    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            news TEXT,
            prediction TEXT,
            confidence REAL,
            date TEXT
        )
    """)

    conn.commit()
    conn.close()


init_db()


# ==========================================
# ARTICLE EXTRACTION FROM URL
# ==========================================

def extract_article(url):

    # Method 1: Trafilatura
    try:
        downloaded = trafilatura.fetch_url(url)

        if downloaded:
            text = trafilatura.extract(
                downloaded,
                include_comments=False,
                include_tables=False
            )

            if text and len(text.strip()) > 100:
                return text.strip()

    except Exception as e:
        print("Trafilatura failed:", e)


    # Method 2: Requests fallback
    try:
        import requests

        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/154.0.0.0 Safari/537.36"
            )
        }

        response = requests.get(
            url,
            headers=headers,
            timeout=15
        )

        response.raise_for_status()

        text = trafilatura.extract(
            response.text,
            include_comments=False,
            include_tables=False
        )

        if text and len(text.strip()) > 100:
            return text.strip()

    except Exception as e:
        print("Requests fallback failed:", e)


    raise Exception("Could not extract article text from this URL")

# ==========================================
# HOME PAGE
# ==========================================

@app.route("/", methods=["GET", "POST"])
def home():

    prediction = None
    confidence = 0
    news = ""
    title = ""

    if request.method == "POST":

        # ==========================================
        # GET TITLE AND NEWS
        # ==========================================

        title = request.form.get("title", "").strip()
        news = request.form.get("news", "").strip()


        # ==========================================
        # CHECK EMPTY INPUT
        # ==========================================

        if not title and not news:

            prediction = "Please enter a title or news article"
            confidence = 0

            return render_template(
                "index.html",
                prediction=prediction,
                confidence=confidence,
                news=news,
                title=title
            )


        # ==========================================
        # CHECK IF INPUT IS A URL
        # ==========================================

        if news.startswith("http://") or news.startswith("https://"):

            try:

                news = extract_article(news)

            except Exception as e:

                print("URL EXTRACTION ERROR:", e)

                prediction = "Could not read this URL"
                confidence = 0

                return render_template(
                    "index.html",
                    prediction=prediction,
                    confidence=confidence,
                    news=news,
                    title=title
                )


        # ==========================================
        # COMBINE TITLE + NEWS
        # ==========================================

        combined_text = title + " " + news


        # ==========================================
        # MACHINE LEARNING PREDICTION
        # ==========================================

        try:

            # Convert text into TF-IDF features
            news_vector = vectorizer.transform([combined_text])


            # Get prediction
            result = model.predict(news_vector)[0]


            # Get probability
            probabilities = model.predict_proba(news_vector)[0]

            confidence = round(max(probabilities) * 100, 2)


            # ==========================================
            # RESULT
            # ==========================================

            if result == 0:

                prediction = "POSSIBLY FAKE"

            else:

                prediction = "LIKELY REAL"


        except Exception as e:

            print("ML PREDICTION ERROR:", e)

            prediction = "Prediction Error"
            confidence = 0


        # ==========================================
        # SAVE RESULT TO HISTORY
        # ==========================================

        conn = sqlite3.connect("history.db")

        cursor = conn.cursor()

        cursor.execute("""
            INSERT INTO history
            (news, prediction, confidence, date)
            VALUES (?, ?, ?, ?)
        """, (
            news,
            prediction,
            confidence,
            datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        ))

        conn.commit()
        conn.close()


    # ==========================================
    # DISPLAY RESULT
    # ==========================================

    return render_template(
        "index.html",
        prediction=prediction,
        confidence=confidence,
        news=news,
        title=title
    )


# ==========================================
# HISTORY PAGE
# ==========================================

@app.route("/history")
def history():

    conn = sqlite3.connect("history.db")

    cursor = conn.cursor()

    cursor.execute("""
        SELECT news, prediction, confidence, date
        FROM history
        WHERE id IN (
            SELECT MAX(id)
            FROM history
            GROUP BY news
        )
        ORDER BY id DESC
    """)

    records = cursor.fetchall()

    conn.close()

    return render_template(
        "history.html",
        records=records
    )


# ==========================================
# CLEAR HISTORY
# ==========================================

@app.route("/clear-history", methods=["POST"])
def clear_history():

    conn = sqlite3.connect("history.db")

    cursor = conn.cursor()

    cursor.execute("DELETE FROM history")

    conn.commit()

    conn.close()

    return redirect("/history")


# ==========================================
# HOW IT WORKS PAGE
# ==========================================

@app.route("/how-it-works")
def how_it_works():

    return render_template("how_it_works.html")


# ==========================================
# ABOUT PAGE
# ==========================================

@app.route("/about")
def about():

    return render_template("about.html")


# ==========================================
# PERFORMANCE PAGE
# ==========================================

@app.route("/performance")
def performance():

    # ==========================================
    # V2 MODEL PERFORMANCE
    # ==========================================

    accuracy = 98.57

    fake_precision = 0.99
    fake_recall = 0.98
    fake_f1 = 0.99
    fake_support = 4696

    real_precision = 0.98
    real_recall = 0.99
    real_f1 = 0.99
    real_support = 4284

    return render_template(
        "performance.html",

        accuracy=accuracy,

        fake_precision=fake_precision,
        fake_recall=fake_recall,
        fake_f1=fake_f1,
        fake_support=fake_support,

        real_precision=real_precision,
        real_recall=real_recall,
        real_f1=real_f1,
        real_support=real_support
    )


# ==========================================
# RUN FLASK APPLICATION
# ==========================================

if __name__ == "__main__":

    app.run(
    host="127.0.0.1",
    port=8765,
    debug=False,
    use_reloader=False
    )