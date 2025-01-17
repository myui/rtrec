FROM debian:bookworm-slim

ENV OPENBLAS_NUM_THREADS=1
ENV DEBCONF_NOWARNINGS=yes
# Enable fault handlers for the SIGSEGV, SIGFPE, SIGABRT, SIGBUS, and SIGILL signals.
# https://docs.python.org/3/library/faulthandler.html
ENV PYTHONFAULTHANDLER=1
ENV PYTHON_VERSION=3.12

RUN apt-get -y update \
 && apt-get install -y git sudo wget curl \
 && apt-get install -y build-essential \
 && apt-get install -y libopenblas-dev \
 && apt-get install -y pkg-config libssl-dev \
 && apt-get install -y libffi-dev \
 && apt-get install -y python3-gdbm \
 && apt-get clean && rm -rf /var/lib/apt/lists/*

RUN groupadd -g 65432 wheel \
 && useradd -u 65432 -g wheel --create-home -s /sbin/nologin td-user \
 && echo '%wheel ALL=NOPASSWD: /usr/bin/apt-get' >> /etc/sudoers

USER td-user
WORKDIR /home/td-user

# Install Rust and Cargo
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal --default-toolchain stable \
  && . ~/.cargo/env \
  && echo '. "$HOME/.cargo/env"' >> ~/.bashrc \
  && rustc --version && cargo --version

# Install UV
RUN curl -LsSf https://astral.sh/uv/install.sh | sh \
 && . ~/.local/bin/env \
 && uv python pin 3.12 \
 && uv venv --python 3.12 \
 && uv pip install streamlit matplotlib jupyter jupyterlab ipywidgets

# workaround for process didn't exit successfully:
# `/home/td-user/rtrec/target/release/build/proc-macro2-27b8d773934653fd/build-script-build` (signal: 11, SIGSEGV: invalid memory reference)
# see https://github.com/rust-lang/cargo/issues/10679#issuecomment-1132612876
RUN git clone https://github.com/myui/rtrec.git && cd rtrec \
  && . ~/.local/bin/env && uv python pin 3.12 && uv venv --python 3.12 && uv sync \
  && uv pip install ipykernel

ENV PYTHONPATH="/home/td-user/rtrec/.venv/lib/python3.12/site-packages"

CMD ["python3"]
