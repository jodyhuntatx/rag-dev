#!/bin/bash
# see: https://www.freecodecamp.org/news/how-to-build-a-chatbot-with-react/
npx create-react-app chatbot
cd chatbot
yarn add react-chatbot-kit sync-fetch
cp ../botsrc/* ./src
