FROM ubuntu:latest

RUN apt-get update && apt-get install -y curl inotify-tools
RUN curl -fsSL https://ollama.com/install.sh | sh
WORKDIR /root
COPY ./docker/ollama/modelfiles/ modelfiles
COPY ./docker/ollama/ollama_scripts/ scripts/
RUN chmod -R +x scripts/
RUN find scripts/ -type f -exec sed -i 's/\r//' {} +
CMD /bin/bash /root/scripts/model_autoloader.sh & ollama serve