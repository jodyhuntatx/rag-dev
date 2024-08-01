class MessageParser {
    constructor(actionProvider) {
      this.actionProvider = actionProvider;
    }
  
    parse(message) {
      const lowerCaseMessage = message.toLowerCase()
      this.actionProvider.submitMessage(lowerCaseMessage)
    }
  }
  
  export default MessageParser
