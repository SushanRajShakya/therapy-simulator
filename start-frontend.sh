#!/bin/bash
# Install dependencies and run Streamlit
conda install -c conda-forge streamlit python-dotenv -y
streamlit run frontend/main.py --server.port 8501