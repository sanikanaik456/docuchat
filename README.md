\# DocuChat — RAG-Powered PDF Question Answering



Upload any PDF and ask questions about it using Google Gemini AI.



\## Features

\- Upload any PDF document

\- Ask natural language questions

\- Answers grounded in real document content (no hallucination!)

\- Built with Python, Streamlit and Google Gemini API



\## Tech Stack

\- Python

\- Streamlit

\- Google Gemini API (google-genai)

\- PyPDF2



\## How to Run

pip install streamlit google-genai PyPDF2

streamlit run app.py



\## How it Works

1\. PDF uploaded and text extracted

2\. Text split into chunks

3\. Relevant chunks found using keyword search

4\. Gemini AI generates grounded answer

