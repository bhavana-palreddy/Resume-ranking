import streamlit as st
import sys
print(sys.executable)

import PyPDF2 
import re
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

# Download NLTK stopwords
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Function to extract text from PDF
def extract_text_from_pdf(pdf_file):
    reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

# Function to preprocess text
def preprocess_text(text):
    text = re.sub(r'\W', ' ', text)  # Remove punctuation
    text = re.sub(r'\s+', ' ', text)  # Remove extra spaces
    text = text.lower()  # Convert to lowercase
    text = ' '.join([word for word in text.split() if word not in stop_words])  # Remove stop words
    return text

# Streamlit App
st.title("AI-Powered Resume screening and Ranking System")
st.write("Upload multiple resumes in PDF format  or in image format(png)and enter a job description to rank the resumes.")

# Upload PDF or png resumes
uploaded_files = st.file_uploader("Upload Resumes (PDF, PNG)", type="pdf png", accept_multiple_files=True)
job_description = st.text_area("Enter Job role's Description")

if uploaded_files and job_description:
    # Extract and preprocess resumes
    resumes = []
    for uploaded_file in uploaded_files:
        text = extract_text_from_pdf(uploaded_file)
        processed_text = preprocess_text(text)
        resumes.append(processed_text)

    # Preprocess job description
    processed_job_description = preprocess_text(job_description)

    # Combine resumes and job description for TF-IDF
    all_texts = resumes + [processed_job_description]

    # TF-IDF Vectorization
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(all_texts)

    # Calculate Cosine Similarity
    cosine_similarities = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1])

    # Rank resumes
    ranked_resumes = sorted(zip(range(len(resumes)), cosine_similarities[0]), key=lambda x: x[1], reverse=True)

    # Display results
    st.subheader("Ranked Resumes")
    results = []
    for idx, score in ranked_resumes:
        results.append({
            "Resume": idx + 1,
            "Similarity Score": f"{score:.4f}",
            "Preview": resumes[idx][:100] + "..."  # Show a preview of the resume
        })

    # Create a DataFrame for better display
    df_results = pd.DataFrame(results)
    st.table(df_results)

    # Option to download results
    if st.button("Download Results as CSV"):
        df_results.to_csv("ranked_resumes.csv", index=False)
        st.success("Results are downloaded as CSV!")
