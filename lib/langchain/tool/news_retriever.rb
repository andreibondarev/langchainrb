# frozen_string_literal: true

module Langchain::Tool
  class NewsRetriever < Base
    #
    # A tool that retrieves latest news from various sources via https://newsapi.org/.
    # An API key needs to be obtained from https://newsapi.org/ to use this tool.
    #
    # Usage:
    #    news_retriever = Langchain::Tool::NewsRetriever.new(api_key: ENV["NEWS_API_KEY"])
    #
    NAME = "news_retriever"
    FUNCTIONS = [:get_everything, :get_top_headlines, :get_sources]

    def initialize(api_key: ENV["NEWS_API_KEY"])
      super()

      @api_key = api_key
    end

    # Retrieve all news
    #
    # @param q [String] Keywords or phrases to search for in the article title and body.
    # @param search_in [String] The fields to restrict your q search to. The possible options are: title, description, content.
    # @param sources [String] A comma-seperated string of identifiers (maximum 20) for the news sources or blogs you want headlines from. Use the /sources endpoint to locate these programmatically or look at the sources index.
    # @param domains [String] A comma-seperated string of domains (eg bbc.co.uk, techcrunch.com, engadget.com) to restrict the search to.
    # @param exclude_domains [String] A comma-seperated string of domains (eg bbc.co.uk, techcrunch.com, engadget.com) to remove from the results.
    # @param from [String] A date and optional time for the oldest article allowed. This should be in ISO 8601 format.
    # @param to [String] A date and optional time for the newest article allowed. This should be in ISO 8601 format.
    # @param language [String] The 2-letter ISO-639-1 code of the language you want to get headlines for. Possible options: ar, de, en, es, fr, he, it, nl, no, pt, ru, se, ud, zh.
    # @param sort_by [String] The order to sort the articles in. Possible options: relevancy, popularity, publishedAt.
    # @param page_size [Integer] The number of results to return per page. 20 is the API's default, 100 is the maximum. Our default is 5.
    # @param page [Integer] Use this to page through the results.
    #
    # @return [String] JSON response
    def get_everything(
      q: nil,
      search_in: nil,
      sources: nil,
      domains: nil,
      exclude_domains: nil,
      from: nil,
      to: nil,
      language: nil,
      sort_by: nil,
      page_size: 5, # The API default is 20 but that's too many.
      page: nil
    )
      Langchain.logger.info("Retrieving all news", for: self.class)

      params = {apiKey: @api_key}
      params[:q] = q if q
      params[:searchIn] = search_in if search_in
      params[:sources] = sources if sources
      params[:domains] = domains if domains
      params[:excludeDomains] = exclude_domains if exclude_domains
      params[:from] = from if from
      params[:to] = to if to
      params[:language] = language if language
      params[:sortBy] = sort_by if sort_by
      params[:pageSize] = page_size if page_size
      params[:page] = page if page

      send_request(path: "everything", params: params)
    end

    # Retrieve top headlines
    #
    # @param country [String] The 2-letter ISO 3166-1 code of the country you want to get headlines for. Possible options: ae, ar, at, au, be, bg, br, ca, ch, cn, co, cu, cz, de, eg, fr, gb, gr, hk, hu, id, ie, il, in, it, jp, kr, lt, lv, ma, mx, my, ng, nl, no, nz, ph, pl, pt, ro, rs, ru, sa, se, sg, si, sk, th, tr, tw, ua, us, ve, za.
    # @param category [String] The category you want to get headlines for. Possible options: business, entertainment, general, health, science, sports, technology.
    # @param sources [String] A comma-seperated string of identifiers for the news sources or blogs you want headlines from. Use the /sources endpoint to locate these programmatically.
    # @param q [String] Keywords or a phrase to search for.
    # @param page_size [Integer] The number of results to return per page. 20 is the API's default, 100 is the maximum. Our default is 5.
    # @param page [Integer] Use this to page through the results.
    #
    # @return [String] JSON response
    def get_top_headlines(
      country: nil,
      category: nil,
      sources: nil,
      q: nil,
      page_size: 5,
      page: nil
    )
      Langchain.logger.info("Retrieving top news headlines", for: self.class)

      params = {apiKey: @api_key}
      params[:country] = country if country
      params[:category] = category if category
      params[:sources] = sources if sources
      params[:q] = q if q
      params[:pageSize] = page_size if page_size
      params[:page] = page if page

      send_request(path: "top-headlines", params: params)
    end

    # Retrieve news sources
    #
    # @param category [String] The category you want to get headlines for. Possible options: business, entertainment, general, health, science, sports, technology.
    # @param language [String] The 2-letter ISO-639-1 code of the language you want to get headlines for. Possible options: ar, de, en, es, fr, he, it, nl, no, pt, ru, se, ud, zh.
    # @param country [String] The 2-letter ISO 3166-1 code of the country you want to get headlines for. Possible options: ae, ar, at, au, be, bg, br, ca, ch, cn, co, cu, cz, de, eg, fr, gb, gr, hk, hu, id, ie, il, in, it, jp, kr, lt, lv, ma, mx, my, ng, nl, no, nz, ph, pl, pt, ro, rs, ru, sa, se, sg, si, sk, th, tr, tw, ua, us, ve, za.
    #
    # @return [String] JSON response
    def get_sources(
      category: nil,
      language: nil,
      country: nil
    )
      Langchain.logger.info("Retrieving news sources", for: self.class)

      params = {apiKey: @api_key}
      params[:country] = country if country
      params[:category] = category if category
      params[:language] = language if language

      send_request(path: "top-headlines/sources", params: params)
    end

    private

    def send_request(path:, params:)
      uri = URI.parse("https://newsapi.org/v2/#{path}?#{URI.encode_www_form(params)}")
      http = Net::HTTP.new(uri.host, uri.port)
      http.use_ssl = true

      request = Net::HTTP::Get.new(uri.request_uri)
      request["Content-Type"] = "application/json"

      response = http.request(request)
      response
        .body
        # Remove non-UTF-8 characters
        .force_encoding(Encoding::UTF_8)
    end
  end
end
