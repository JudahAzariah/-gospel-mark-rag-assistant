# AI-Powered RAG Application: Gospel of Mark

**Graduate Coursework Project**  
**CSCI 5650: Professional AI-Based Software Architectures (from Startup to Enterprise)**  
**California Baptist University**

This project is a Retrieval-Augmented Generation (RAG) application developed as part of my graduate coursework at California Baptist University.

The application allows users to ask questions about the Gospel of Mark and generates answers based on relevant passages retrieved from the source material. The project focuses on building a complete AI application, from retrieval and LLM integration to evaluation, deployment, guardrails, and monitoring.

## Project Overview

The main goal of this project was to build a practical RAG system that can retrieve relevant information from a source document and use an LLM to generate an answer based on that information.

The application uses the Gospel of Mark as its knowledge source. Instead of sending the entire document to the LLM for every question, the system first searches for relevant passages and then provides those passages as context for the response.

This helps keep responses focused on the available source material and reduces the chance of unsupported answers.

## Architecture

The application follows a basic RAG pipeline:

```text
User Question
      |
      v
Embedding Generation
      |
      v
Vector Search
      |
      v
Relevant Passages
      |
      v
LLM with Retrieved Context
      |
      v
Generated Response
```

The main components are:

- Document corpus containing the Gospel of Mark
- Text chunking and preprocessing
- Sentence Transformer embeddings
- Vector similarity search
- Retrieval of relevant passages
- LLM-based response generation
- Evaluation and comparison of different configurations
- Request logging and monitoring
- Rate limiting and error handling
- Streamlit-based web interface

## Technology Stack

**Programming**
- Python

**AI and Machine Learning**
- Large Language Models
- Retrieval-Augmented Generation (RAG)
- Sentence Transformers
- Vector embeddings
- Semantic search

**Libraries and Tools**
- Streamlit
- NumPy
- Python dotenv
- Google Gemini API

**Development**
- Git
- GitHub
- Virtual environments
- Environment variables

## Evaluation

The project includes an evaluation pipeline to test the quality of the RAG system.

Different retrieval configurations were tested by changing parameters such as:

- Chunk size
- Chunk overlap
- Number of retrieved passages
- Retrieval settings

The evaluation system records individual results as well as overall performance. Failure analysis was also used to identify common types of errors and areas where the system could be improved.

## Reliability and Observability

The project also includes several features intended to make the application more reliable outside of a basic prototype.

These include:

- Request logging
- Response latency tracking
- Token usage tracking
- Estimated request cost
- Rate limiting
- API failure handling
- Graceful error messages
- Usage summaries

The application records request information so that performance and usage can be reviewed after deployment.

## Deployment

The application is deployed using Streamlit Community Cloud and can be accessed publicly.

**Live Application:**  
https://pbfwa5vwaau9tjgqyovwbo.streamlit.app/

The application uses environment variables for API configuration so that API keys are not stored in the GitHub repository.

## Project Structure

```text
.
├── main.py
├── streamlit_app.py
├── evaluate_lab3.py
├── failure_analysis.py
├── compare_configs.py
├── test_set.json
├── evaluation_results.md
├── requirements.txt
│
├── configs/
│   ├── baseline.json
│   ├── chunk_300.json
│   ├── topk_2.json
│   └── topk_5.json
│
├── results/
│   ├── comparison.json
│   └── comparison.md
│
├── logs/
│   └── requests.jsonl
│
└── corpus/
    └── mark.txt
```

## Running the Project Locally

Clone the repository and install the required packages:

```bash
git clone <repository-url>
cd <repository-folder>
pip install -r requirements.txt
```

Create a `.env` file based on `.env.example` and add the required API configuration.

Then start the application:

```bash
python -m streamlit run streamlit_app.py
```

The application will be available at:

```text
http://localhost:8501
```

## Academic Context

This project was developed for graduate coursework at California Baptist University as part of CSCI 5650.

The project covers concepts including:

- Retrieval-Augmented Generation
- LLM and API integration
- Prompt design
- Vector search
- AI evaluation
- Cost and latency analysis
- Application deployment
- Rate limiting
- Error handling
- Logging and observability

The project was developed incrementally through the course, starting with a basic RAG application and extending it with evaluation, optimization, deployment, and operational features.

## What I Learned

This project gave me hands-on experience building an AI application beyond a simple chatbot. It involved working with retrieval systems, LLM APIs, evaluation methods, deployment platforms, and the operational issues that come with running an AI application.

The main focus was not just getting an LLM to generate an answer, but understanding how the different parts of an AI system work together in a complete application.
