#!/bin/sh

if [ -z "${OMDB_API_KEY}" ]; then
    echo "enviroment variable OMDB_API_KEY is required. \nexport OMDB_API_KEY=your_api_key"
    exit 1
fi

ARCH=$(uname -m)
PROJECT_DIR=$(cd $(dirname $0); cd ..; pwd)

# -v ${PROJECT_DIR}/examples/streamlit:/home/td-user/rtrec/examples/streamlit
docker run --platform=linux/${ARCH} -it --rm -p 8501:8501 --env OMDB_API_KEY myui/rtrec:${ARCH} \
bash -c "cd rtrec && git fetch && git pull && . ~/.local/bin/env && uv sync && uv pip install streamlit && .venv/bin/streamlit run examples/streamlit/movielens/movielens_dashboard.py --server.port=8501 --server.address=0.0.0.0"
