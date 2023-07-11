# frozen_string_literal: true

module Langchain::Vectorsearch
  class Elasticsearch < Base
    attr_accessor :es_client, :index_name

    def initialize(url:, index_name:, llm:, api_key: nil)
      require "elastisearch"

      @es_client = Elasticsearch::Client.new(
        url: url,
        request_timeout: 20,
        log: false,
      )
      @index_name = index_name

      super(llm: llm)
    end

    def create_default_schema
      es_client.client.indices.create(
        index: index_name,
        body: default_schema
      )
    end

    def default_schema
      {
        mappings: {
          properties: {
            title: {
              type: "text"
            },
            embedding: {
              type: "dense_vector",
              dims: 384
            }
          }
        }
      }
    end
  end
end
