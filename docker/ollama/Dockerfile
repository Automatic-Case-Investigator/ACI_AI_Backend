FROM ubuntu:latest

RUN apt-get update && apt-get install -y curl python3 python3-pip
RUN ln -s /usr/bin/python3 /usr/bin/python
RUN curl -fsSL https://ollama.com/install.sh | sh
WORKDIR /root
COPY ./docker/ollama/modelfiles/ modelfiles
COPY ./docker/ollama/ollama_scripts/ scripts/
RUN find scripts/ -type f -exec sed -i 's/\r//' {} +
CMD python3 /root/scripts/model_autoloader.py & ollama serve