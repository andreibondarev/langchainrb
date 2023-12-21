# frozen_string_literal: true

module Langchain
  # Abstraction for data loaded by a {Langchain::Loader}
  class Data
    # URL or Path of the data source
    # @return [String]
    attr_reader :source

    # @param data [String] data that was loaded
    # @option options [String] :source URL or Path of the data source
    def initialize(data, options = {})
      @source = options[:source]
      @data = data
      @chunker_klass = options[:chunker] || Langchain::Chunker::Text
    end

    # @return [String]
    def value
      @data
    end

    # @param opts [Hash] options passed to the chunker
    # @return [Array<String>]
    def chunks(opts = {})
      @chunker_klass.new(@data, **opts).chunks
    end
  end
end
