#!/bin/bash
PS_OUT=$(ps -aux | grep start.js | grep -v grep)
if [[ "$PS_OUT" == "" ]]; then
  echo "No react server found running."
else
  echo $PS_OUT
fi
echo