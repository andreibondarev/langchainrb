# frozen_string_literal: true

module Langchain::Tool
  class Calculator < Base
    #
    # A calculator tool that falls back to the Google calculator widget
    #
    # Gem requirements:
    #   gem "eqn", "~> 1.6.5"
    #   gem "google_search_results", "~> 2.0.0"
    #

    NAME = "calculator"

    description <<~DESC
      Useful for getting the result of a math expression.

      The input to this tool should be a valid mathematical expression that could be executed by a simple calculator.
    DESC

    def initialize
      depends_on "eqn"
      require "eqn"
    end

    # Evaluates a pure math expression or if equation contains non-math characters (e.g.: "12F in Celsius") then
    # it uses the google search calculator to evaluate the expression
    # @param input [String] math expression
    # @return [String] Answer
    def execute(input:)
      Langchain.logger.info("Executing \"#{input}\"", for: self.class)

      Eqn::Calculator.calc(input)
    rescue Eqn::ParseError, Eqn::NoVariableValueError
      # Sometimes the input is not a pure math expression, e.g: "12F in Celsius"
      # We can use the google answer box to evaluate this expression
      # TODO: Figure out to find a better way to evaluate these language expressions.
      hash_results = Langchain::Tool::GoogleSearch
        .new(api_key: ENV["SERPAPI_API_KEY"])
        .execute_search(input: input)
      hash_results.dig(:answer_box, :to) ||
        hash_results.dig(:answer_box, :result)
    end
  end
end
