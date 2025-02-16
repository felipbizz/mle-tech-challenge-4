#!/bin/bash

dir=".logfire"

if [ ! -d "$dir" ]; then
  mkdir "$dir"
  touch "$dir/logfire_credentials.json"
  echo "Diretório $dir criado. Arquivo logfire_credentials.json vazio criado."
else
  echo "Diretório $dir já existe."
fi
