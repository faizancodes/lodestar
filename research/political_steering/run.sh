#!/bin/sh
# send a python snippet (file or stdin) to the running repl.py
curl -s --max-time 3000 -X POST --data-binary @"${1:--}" 127.0.0.1:8799/run
