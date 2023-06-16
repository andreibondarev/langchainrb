# frozen_string_literal: true

require "strscan"
require "json"
require "yaml"

module Langchain::Prompt
  # Prompts are structured inputs to the LLMs. Prompts provide instructions, context and other user input that LLMs use to generate responses.
  #
  # @abstract
  class Base
    def format(**kwargs)
      raise NotImplementedError
    end

    # @return [String] the type of the prompt
    def prompt_type
      raise NotImplementedError
    end

    # @return [Hash] a hash representation of the prompt
    def to_h
      raise NotImplementedError
    end

    #
    # Validate the input variables against the template.
    #
    # @param template [String] The template to validate against.
    # @param input_variables [Array<String>] The input variables to validate.
    #
    # @raise [ArgumentError] If there are missing or extra variables.
    #
    # @return [void]
    #
    def validate(template:, input_variables:)
      input_variables_set = @input_variables.uniq
      variables_from_template = Langchain::Prompt::Base.extract_variables_from_template(template)

      missing_variables = variables_from_template - input_variables_set
      extra_variables = input_variables_set - variables_from_template

      raise ArgumentError, "Missing variables: #{missing_variables}" if missing_variables.any?
      raise ArgumentError, "Extra variables: #{extra_variables}" if extra_variables.any?
    end

    #
    # Save the object to a file in JSON or YAML format.
    #
    # @param file_path [String, Pathname] The path to the file to save the object to
    #
    # @raise [ArgumentError] If file_path doesn't end with .json or .yaml or .yml
    #
    # @return [void]
    #
    def save(file_path:)
      save_path = file_path.is_a?(String) ? Pathname.new(file_path) : file_path
      directory_path = save_path.dirname
      FileUtils.mkdir_p(directory_path) unless directory_path.directory?

      case save_path.extname
      when ".json"
        File.write(file_path, to_h.to_json)
      when ".yaml", ".yml"
        File.write(file_path, to_h.to_yaml)
      else
        raise ArgumentError, "#{file_path} must be json or yaml file"
      end
    end

    #
    # Extracts variables from a template string.
    #
    # This method takes a template string and returns an array of input variable names
    # contained within the template. Input variables are defined as text enclosed in
    # curly braces (e.g. <code>\{variable_name\}</code>).
    #
    # Content within two consecutive curly braces (e.g. <code>\{\{ignore_me}}</code>) are ignored.
    #
    # @param template [String] The template string to extract variables from.
    #
    # @return [Array<String>] An array of input variable names.
    #
    def self.extract_variables_from_template(template)
      input_variables = []
      scanner = StringScanner.new(template)

      while scanner.scan_until(/\{([^}]*)\}/)
        variable = scanner[1].strip
        input_variables << variable unless variable.empty? || variable[0] == "{"
      end

      input_variables
    end
  end
end
