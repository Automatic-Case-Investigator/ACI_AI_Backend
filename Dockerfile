FROM ubuntu:latest

# Configures ollama
WORKDIR /app
RUN curl -fsSL https://ollama.com/install.sh | sh
EXPOSE 11434

# Installs python
WORKDIR /app
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y \
    software-properties-common \
    wget \
    build-essential \
    libssl-dev \
    zlib1g-dev \
    libbz2-dev \
    libreadline-dev \
    libsqlite3-dev \
    curl \
    libncurses5-dev \
    libncursesw5-dev \
    xz-utils \
    tk-dev \
    libffi-dev \
    liblzma-dev \
    git \
    && apt-get clean

RUN add-apt-repository ppa:deadsnakes/ppa && apt-get update
RUN apt-get install -y python3.12 python3.12-venv python3.12-dev
RUN python3.12 -m venv venv
ENV VIRTUAL_ENV=venv
ENV PATH="venv/bin:$PATH"
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.12

# Configures the django backend
ENV PYTHONUNBUFFERED=1
WORKDIR /app
COPY requirements.txt /app
RUN pip install --no-cache-dir -r requirements.txt
COPY . /app
EXPOSE 8000

CMD python3 manage.py runserver 0.0.0.0:8000