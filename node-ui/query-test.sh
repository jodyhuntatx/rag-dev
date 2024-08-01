#!/bin/bash -x
#bot_response=$(
curl -X POST http://99.79.38.81:8000/query 		\
	-H 'Accept: application/json' 			\
	-H 'Content-Type: application/json'		\
	-d "{\"data\": \"$1\"}"                     	\
#)
#printf '%s' "$bot_response"
