#!/bin/bash
PID=$(ps -aux | grep start.js | grep -v grep | awk '{print $2}')
if [[ "$PID" == "" ]]; then
  echo "No react server found running."
else
  kill -2 $PID
  echo "UI stopped"
fi