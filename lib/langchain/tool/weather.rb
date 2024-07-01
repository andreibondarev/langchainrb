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
    #     weather.execute(input: "Boston, MA; imperial")
    #
    NAME = "weather"
    FUNCTIONS = [:execute]

    attr_reader :client, :units

    # Initializes the Weather tool
    #
    # @param api_key [String] Open Weather API key
    # @return [Langchain::Tool::Weather] Weather tool
    def initialize(api_key:, units: "metric")
      super()

      depends_on "open-weather-ruby-client"
      require "open-weather-ruby-client"

      OpenWeather::Client.configure do |config|
        config.api_key = api_key
        config.user_agent = "Langchainrb Ruby Client"
      end

      @client = OpenWeather::Client.new
    end

    # Returns current weather for a city
    #
    # @param input [String] comma separated city and unit (optional: imperial, metric, or standard)
    # @return [String] Answer
    def execute(input:)
      Langchain.logger.info("Executing for \"#{input}\"", for: self.class)

      input_array = input.split(";")
      city, units = *input_array.map(&:strip)

      data = client.current_weather(city: city, units: units)
      weather = data.main.map { |key, value| "#{key} #{value}" }.join(", ")
      "The current weather in #{data.name} is #{weather}"
    end
  end
end
