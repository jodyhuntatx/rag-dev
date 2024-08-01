
// see: https://www.npmjs.com/package/sync-fetch
const fetch = require('sync-fetch')

const API_BASE_URL = 'http://ec2-99-79-38-81.ca-central-1.compute.amazonaws.com:8000/query';

class ActionProvider {
    constructor(createChatBotMessage, setStateFunc) {
      this.createChatBotMessage = createChatBotMessage;
      this.setState = setStateFunc;
    }

    submitMessage(message) {
      const options = {
        method: 'POST',
        headers: {
          'Accept': '*/*',
          'Content-Type': 'application/json',
        },
        body: `{"data": "${message}"}`
      };
      const response = fetch(API_BASE_URL, options).text();
      const botMessage = this.createChatBotMessage(util.format('%s', response));
      this.updateChatbotState(botMessage);
    }

    updateChatbotState(message) {
    // NOTE: This function is set in the constructor, and is passed in
    // from the top level Chatbot component. The setState function here
    // actually manipulates the top level state of the Chatbot, so it's
    // important that we make sure that we preserve the previous state.

     this.setState(prevState => ({
          ...prevState, messages: [...prevState.messages, message]
      }))
    }
  }

  export default ActionProvider