#!/bin/bash

docker compose up --detach --build
bash ./request-script.sh
bash ./trace.sh
