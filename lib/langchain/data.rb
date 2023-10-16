# frozen_string_literal: true

module Langchain
  # Abstraction for data loaded by a {Langchain::Loader}
  class Data
    attr_reader :source, :source_type

    # @param data [String] data that was loaded
    # @option options [String] :source URL or Path of the data source
    # @option options [String] :source_type type of the data source
    def initialize(data, options = {})
      @data = data
      @source = options[:source]
      @source_type = options[:source_type]
    end

    # @return [String]
    def value
      @data
    end

    # @param options [Hash] options passed to the chunker
    # @return [Array<String>]
    def chunks(options = {})
      if Langchain::Processors::Code::EXTENSIONS.include?(source_type)
        options = options.merge(separators: Langchain::Chunker::Base::LANG_SEPARATORS[source_type])
        Langchain::Chunker::RecursiveText.new(@data, **options).chunks
      else
        Langchain::Chunker::Text.new(@data, **options).chunks
      end
    end
  end
end
