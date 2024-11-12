FROM debian:bookworm-slim

ENV OPENBLAS_NUM_THREADS=1
ENV DEBCONF_NOWARNINGS=yes
# Enable fault handlers for the SIGSEGV, SIGFPE, SIGABRT, SIGBUS, and SIGILL signals.
# https://docs.python.org/3/library/faulthandler.html
ENV PYTHONFAULTHANDLER=1

RUN apt-get -y update \
 && apt-get install -y git sudo wget curl \
 && apt-get install -y build-essential \
 && apt-get install -y libopenblas-dev \
 && apt-get install -y pkg-config libssl-dev \
 && apt-get install -y libffi-dev \
 && apt-get clean && rm -rf /var/lib/apt/lists/*

RUN groupadd -g 65432 wheel \
 && useradd -u 65432 -g wheel --create-home -s /sbin/nologin td-user \
 && echo '%wheel ALL=NOPASSWD: /usr/bin/apt-get' >> /etc/sudoers

USER td-user

# Install Rust and Cargo
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal --default-toolchain stable

# Install Rye and maturin
RUN curl -sSf https://rye.astral.sh/get | RYE_NO_AUTO_INSTALL=1 RYE_INSTALL_OPTION="--yes" bash \
 && . "$HOME/.rye/env" \
 && echo '. "$HOME/.rye/env"' >> "$HOME/.bashrc" \
 && rye install maturin \
 && rye install streamlit && rye install matplotlib \
 && rye install jupyter && rye install jupyterlab && rye install ipywidgets

#RUN mkdir -p /home/td-user/build
#COPY --chown=td-user:wheel README.md pyproject.toml Cargo.toml requirements.lock requirements-dev.lock .python-version /home/td-user/build/
#COPY --chown=td-user:wheel src/ /home/td-user/build/src/
#COPY --chown=td-user:wheel rtrec/ /home/td-user/build/rtrec/
#RUN . "$HOME/.cargo/env" && . "$HOME/.rye/env" && cd /home/td-user/build/ && rye sync --no-lock

ENV PYTHONPATH="/home/td-user/rtrec/.venv/lib/python3.12/site-packages"

WORKDIR /home/td-user

CMD ["python3"]
