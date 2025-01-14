import { createChatBotMessage } from 'react-chatbot-kit';

const config = { 
  botName: "ConsultaBot",
  initialMessages: [createChatBotMessage("Hi, I'm here to help. What do you want to know about Conjur Cloud?")],
  customStyles: {
    botMessageBox: {
      backgroundColor: "#376B7E",
    },
    chatButton: {
      backgroundColor: "#376B7E",
    },
  },
}

export default config