# frozen_string_literal: true

module Langchain::Tool
  class Calculator < Base
    #
    # A calculator tool that falls back to the Google calculator widget
    #
    # Gem requirements:
    #     gem "eqn", "~> 1.6.5"
    #     gem "google_search_results", "~> 2.0.0"
    #
    # Usage:
    #     calculator = Langchain::Tool::Calculator.new
    #
    NAME = "calculator"
    FUNCTIONS = [:execute]

    def initialize
      super()

      depends_on "eqn"
    end

    # Evaluates a pure math expression or if equation contains non-math characters (e.g.: "12F in Celsius") then it uses the google search calculator to evaluate the expression
    #
    # @param input [String] math expression
    # @return [String] Answer
    def execute(input:)
      Langchain.logger.info("Executing \"#{input}\"", for: self.class)

      Eqn::Calculator.calc(input)
    rescue Eqn::ParseError, Eqn::NoVariableValueError
      "\"#{input}\" is an invalid mathematical expression"
    end
  end
end
