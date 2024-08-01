#!/bin/bash
uvicorn consultabot-api:app             \
        --host 0.0.0.0 --port 8000      \
        --log-level debug               \
        >> logs/consultabot.log 2>&1 &
