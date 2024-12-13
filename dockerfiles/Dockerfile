FROM debian:bookworm-slim

ENV OPENBLAS_NUM_THREADS=1
ENV DEBCONF_NOWARNINGS=yes
# Enable fault handlers for the SIGSEGV, SIGFPE, SIGABRT, SIGBUS, and SIGILL signals.
# https://docs.python.org/3/library/faulthandler.html
ENV PYTHONFAULTHANDLER=1
ENV PYTHON_VERSION=3.12
ENV NODE_MAJOR=20

RUN apt-get -y update \
 && apt-get install -y git sudo wget curl \
 && apt-get install -y build-essential \
 && apt-get install -y libopenblas-dev \
 && apt-get install -y pkg-config libssl-dev \
 && apt-get install -y libffi-dev \
 && apt-get install -y python3-gdbm \
 # Node.js >=20 is required for JupyterLab
# && curl -fsSL https://deb.nodesource.com/setup_${NODE_MAJOR}.x | bash - && apt-get install -y nodejs \
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

# Install Rye and maturin
RUN curl -sSf https://rye.astral.sh/get | RYE_NO_AUTO_INSTALL=1 RYE_INSTALL_OPTION="--yes" bash \
 && . ~/.rye/env \
 && echo '. "$HOME/.rye/env"' >> ~/.bashrc \
 && rye pin 3.12 \
 && rye install maturin \
 && rye install streamlit && rye install matplotlib \
 && rye install jupyter && rye install jupyterlab && rye install ipywidgets

# workaround for process didn't exit successfully:
# `/home/td-user/rtrec/target/release/build/proc-macro2-27b8d773934653fd/build-script-build` (signal: 11, SIGSEGV: invalid memory reference)
# see https://github.com/rust-lang/cargo/issues/10679#issuecomment-1132612876
RUN git clone https://github.com/myui/rtrec.git && cd rtrec \
  && . ~/.cargo/env && cargo update && cargo clean && . ~/.rye/env && rye pin 3.12 \
  && RUSTFLAGS="-Ctarget-feature=-crt-static" RUST_BACKTRACE=1 rye sync

ENV PYTHONPATH="/home/td-user/rtrec/.venv/lib/python3.12/site-packages"

CMD ["python3"]
