cd tc4-api

dir=".logfire"

if [ ! -d "$dir" ]; then
  mkdir "$dir"
  touch "$dir/logfire_credentials.json"
  echo "Diretório $dir criado. Arquivo logfire_credentials.json vazio criado."
else
  echo "Diretório $dir já existe."
fi

docker build -f Dockerfile -t mle-api --secret id=logfire,src=.logfire/logfire_credentials.json .

cd ..

sudo chmod -R 777 volumes/
docker-compose up -d --build