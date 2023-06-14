# frozen_string_literal: true

module Langchain::Agent
  # = Agents
  #
  # Agents are semi-autonomous bots that can respond to user questions and use available to them Tools to provide informed replies. They break down problems into series of steps and define Actions (and Action Inputs) along the way that are executed and fed back to them as additional information. Once an Agent decides that it has the Final Answer it responds with it.
  #
  # Available:
  # - {Langchain::Agent::ChainOfThoughtAgent}
  #
  # @abstract
  class Base
    def self.logger_options
      {
        color: :red
      }
    end
  end
end
