# frozen_string_literal: true

module Langchain::Tool
  class Weather < Base
    #
    # A weather tool that gets current weather data
    #
    # Current weather data is free for 1000 calls per day (https://home.openweathermap.org/api_keys)
    # Forecast and historical data require registration with credit card, so not supported yet.
    #
    # Gem requirements:
    #     gem "open-weather-ruby-client", "~> 0.3.0"
    #     api_key: https://home.openweathermap.org/api_keys
    #
    # Usage:
    #     weather = Langchain::Tool::Weather.new(api_key: ENV["OPEN_WEATHER_API_KEY"])
    #
    #     # Examples:
    #     weather.current_weather(city: "Boston")
    #     weather.current_weather(city: "Los Angeles, CA", units: "imperial")
    #     weather.current_weather(city: "Rome, IT", units: "metric")
    #
    NAME = "weather"
    ANNOTATIONS_PATH = Langchain.root.join("./langchain/tool/#{NAME}/#{NAME}.json").to_path

    attr_reader :client

    # Initializes the Weather tool
    #
    # @param api_key [String] Open Weather API key
    # @return [Langchain::Tool::Weather] Weather tool
    def initialize(api_key:)
      depends_on "open-weather-ruby-client"
      require "open-weather-ruby-client"

      OpenWeather::Client.configure do |config|
        config.api_key = api_key
      end

      @client = OpenWeather::Client.new
    end

    # Returns current weather for a city
    # @param city [String] City name, optional ISO 3166 state code (USA only), and optional ISO 3166 country code divided by comma.
    # @param units [String] Units for response. One of "standard", "metric", or "imperial".
    # @return [String] Description of the weather
    def current_weather(city:, units: "standard")
      Langchain.logger.info("Executing current_weather for #{city} in #{units}", for: self.class)

      begin
        weather_data = @client.current_weather(city: city, units: units)
      rescue Faraday::ResourceNotFound
        return "Sorry, I couldn't find the weather for #{city}"
      end

      weather = weather_data.main.map { |key, value| "#{key} #{value}" }.join(", ")
      "The current weather in #{weather_data.name} in #{units} units is #{weather}"
    end
  end
end
