#!/bin/sh

if [ -z "${OMDB_API_KEY}" ]; then
    echo "enviroment variable OMDB_API_KEY is required. \nexport OMDB_API_KEY=your_api_key"
    exit 1
fi

ARCH=$(uname -m)
PROJECT_DIR=$(cd $(dirname $0); cd ..; pwd)

docker run --platform=linux/${ARCH} -it --rm -p 8501:8501 --env OMDB_API_KEY -v ${PROJECT_DIR}:/home/td-user/rtrec myui/rtrec:${ARCH} \
bash -c "cd rtrec && source ~/.cargo/env && ~/.rye/shims/rye sync && ~/.rye/shims/rye remove rtrec && export RUST_LOG=debug && ~/.rye/shims/streamlit run examples/streamlit/movielens_dashboard.py --server.port=8501 --server.address=0.0.0.0"
