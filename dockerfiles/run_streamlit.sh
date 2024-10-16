#!/bin/sh

if [ -z "${OMDB_API_KEY}" ]; then
    echo "enviroment variable OMDB_API_KEY is required. \nexport OMDB_API_KEY=your_api_key"
    exit 1
fi

docker run --platform=linux/amd64 -it --rm -p 8501:8501 --env OMDB_API_KEY -v /Users/myui/workspace/myui/rtrec:/home/td-user/rtrec myui/rtrec:0.1 \
streamlit run rtrec/samples/streamlit/movielens_dashboard.py --server.port=8501 --server.address=0.0.0.0
