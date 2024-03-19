# frozen_string_literal: true

# RubyCodeInterpreter does not work with Ruby 3.3;
# https://github.com/ukutaht/safe_ruby/issues/4
if RUBY_VERSION <= "3.2.0"
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
      ANNOTATIONS_PATH = Langchain.root.join("./langchain/tool/#{NAME}/#{NAME}.json").to_path

      description <<~DESC
        A Ruby code interpreter. Use this to execute ruby expressions. Input should be a valid ruby expression. If you want to see the output of the tool, make sure to return a value.
      DESC

      def initialize(timeout: 30)
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
end
