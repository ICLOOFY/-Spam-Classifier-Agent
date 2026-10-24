'''Spam Classifier Agent - hosted on Agentverse.'''

from uagents import Agent

from protocols import chat_protocol, classifier_protocol

agent = Agent()
agent.include(chat_protocol, publish_manifest=True)
agent.include(classifier_protocol, publish_manifest=True)

if __name__ == "__main__":
    agent.run()