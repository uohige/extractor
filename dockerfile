FROM python:3.11-bookworm

RUN apt-get update && apt-get install -y \
    iproute2 \
    iputils-ping \
    vim \
    less \
    psmisc \
    libgl1 \
    poppler-utils \
    tesseract-ocr \
    libtesseract-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
    unstructured[csv,doc,docx,md,pdf,ppt,pptx,tsv,xlsx] \
    numpy \
    pandas \
    openai \
    tiktoken \
    faiss-cpu \
    loguru \
    python-dotenv \
    streamlit

CMD ["/bin/bash"]
