# frozen_string_literal: true

module Langchain::Tool
  class Tavily < Base
    #
    # Tavily Search is a robust search API tailored specifically for LLM Agents.
    # It seamlessly integrates with diverse data sources to ensure a superior, relevant search experience.
    #
    # Usage:
    #    tavily = Langchain::Tool::Tavily.new(api_key: ENV["TAVILY_API_KEY"])
    #
    NAME = "tavily"
    FUNCTIONS = [:search]

    def initialize(api_key:)
      super()

      @api_key = api_key
    end

    # Search for data based on a query.
    #
    # @param query [String] The search query string.
    # @param search_depth [String] The depth of the search. It can be basic or advanced. Default is basic for quick results and advanced for indepth high quality results but longer response time. Advanced calls equals 2 requests.
    # @param include_images [Boolean] Include a list of query related images in the response. Default is False.
    # @param include_answer [Boolean] Include answers in the search results. Default is False.
    # @param include_raw_content [Boolean] Include raw content in the search results. Default is False.
    # @param max_results [Integer] The number of maximum search results to return. Default is 5.
    # @param include_domains [Array<String>] A list of domains to specifically include in the search results. Default is None, which includes all domains.
    # @param exclude_domains [Array<String>] A list of domains to specifically exclude from the search results. Default is None, which doesn't exclude any domains.
    #
    # @return [String] The search results in JSON format.
    def search(
      query:,
      search_depth: "basic",
      include_images: false,
      include_answer: false,
      include_raw_content: false,
      max_results: 5,
      include_domains: [],
      exclude_domains: []
    )
      uri = URI("https://api.tavily.com/search")
      request = Net::HTTP::Post.new(uri)
      request.content_type = "application/json"
      request.body = {
        api_key: @api_key,
        query: query,
        search_depth: search_depth,
        include_images: include_images,
        include_answer: include_answer,
        include_raw_content: include_raw_content,
        max_results: max_results,
        include_domains: include_domains,
        exclude_domains: exclude_domains
      }.to_json

      response = Net::HTTP.start(uri.hostname, uri.port, use_ssl: uri.scheme == "https") do |http|
        http.request(request)
      end
      response.body
    end
  end
end
