# frozen_string_literal: true

module Langchain::Tool
  class RubyCodeInterpreter < Base
    #
    # A tool that execute Ruby code in a sandboxed environment.
    #
    # Gem requirements:
    #     gem "safe_ruby", "~> 1.0.4"
    #
    # Usage:
    #    interpreter = Langchain::Tool::RubyCodeInterpreter.new
    #
    NAME = "ruby_code_interpreter"
    FUNCTIONS = [:execute]

    def initialize(timeout: 30)
      super()

      depends_on "safe_ruby"

      @timeout = timeout
    end

    # Executes Ruby code in a sandboxes environment.
    #
    # @param input [String] ruby code expression
    # @return [String] Answer
    def execute(input:)
      Langchain.logger.info("Executing \"#{input}\"", for: self.class)

      safe_eval(input)
    end

    def safe_eval(code)
      SafeRuby.eval(code, timeout: @timeout)
    end
  end
end
