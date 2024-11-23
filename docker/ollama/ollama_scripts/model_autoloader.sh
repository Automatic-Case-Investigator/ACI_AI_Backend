#!/bin/bash

directory_to_watch="/root/models"

extension=".gguf"

find $directory_to_watch -type f -name "*.gguf" | while read -r file; do
  file_name="${file%.*}"
  file_name="$(basename $file_name)"
  ollama create $file_name -f "/root/modelfiles/$file_name"
done

while true; do
  inotifywait -m -r -e modify,create,delete "$directory_to_watch" |
    while read path action file; do
      if [[ "$action" == "CREATE" || "$action" == "MODIFY" ]]; then
        if [[ "$file" == *"$extension" ]]; then
          echo "Handling model $action"
          file_name="${file%.*}"
          ollama create $file_name -f "/root/modelfiles/$file_name"
        fi
      fi
      if [[ "$action" == "DELETE" ]]; then
        if [[ "$file" == *"$extension" ]]; then
          echo "Handling model $action"
          file_name="${file%.*}"
          ollama rm $file_name
        fi
      fi
    done
done