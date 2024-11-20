#!/bin/bash

directory_to_watch="/root/models"

extension=".gguf"

while true; do
  inotifywait -m -r -e modify,create,delete "$directory_to_watch" |
    while read path action file; do
      if [[ "$file" == *"$extension_to_watch" ]]; then
        dir_name=$(dirname "$file")
        base_name=$(basename "$file")
        file_name="${base_name%.*}"
        ollama create -f "/root/modelfiles/$file_name"
      fi
    done
done