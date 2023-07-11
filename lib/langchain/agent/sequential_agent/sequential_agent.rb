require "json"

module Langchain::Agent
  class SequentialAgent < Base
    attr_accessor :tools

    # Initializes the Agent
    #
    # @param llm [Object] The LLM client to use
    # @param tools [Array] The tools to use
    # @return [ChainOfThoughtAgent] The Agent::SequentialAgent instance
    def initialize(llm:, tools:)
      # NOTE: don't need to validate tools here. For sequential agent it's fine to have duplicate tools

      @tools = tools || []
    end

    # Run the Agent!
    #
    # @param question [String] The question to ask
    # @return [String] The answer to the question
    def run(question:)
      current_input = question

      Langchain.logger.info("Starting [#{self.class.name}]. Current input is:\n#{current_input}", for: self.class)
      output = ""

      tools.each_with_index do |tool, i|
        tool_result = tool.execute(input: current_input).to_s.strip

        output = tool_result

        current_input = output.strip

        Langchain.logger.info("[Tool #{tool.class.name}] returns output:\n#{current_input}", for: tool.class)
      end

      output
    end
  end
end
