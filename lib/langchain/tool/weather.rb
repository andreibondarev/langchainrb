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
    #     weather = Langchain::Tool::Weather.new(api_key: "YOUR_API_KEY")
    #     weather.execute(input: "Boston, MA; imperial")
    #

    NAME = "weather"

    description <<~DESC
      Useful for getting current weather data

      The input to this tool should be a city name followed by the units (imperial, metric, or standard).
      The Observation temperature will be in Fahrenheit, Celsius, or Kelvin respectively.

      Usage:
        Action Input: St Louis, Missouri; metric
        Action Input: Boston, Massachusetts; imperial
        Action Input: Dubai, AE; imperial
        Action Input: Kiev, Ukraine; metric
    DESC

    #
    # Initializes the Weather tool
    #
    # @param api_key [String] Open Weather API key
    # @return [Langchain::Tool::Weather] Weather tool
    #
    def initialize(api_key:)
      depends_on "open-weather-ruby-client"
      require "open-weather-ruby-client"

      OpenWeather::Client.configure do |config|
        config.api_key = api_key
        config.user_agent = "Langchainrb Ruby Client"
      end

      @client = OpenWeather::Client.new
    end

    # Returns current weather for a city
    # @param input [String] semicolon separated city and unit (optional: imperial, metric, or standard)
    # @return [String] Answer
    def execute(input:)
      Langchain.logger.info("Executing for \"#{input}\"", for: self.class)

      input_array = input.split(";")
      city, units = *input_array.map(&:strip)
      units = "standard" if units.nil?

      begin
        data = @client.current_weather(city: city, units: units)
      rescue Faraday::ResourceNotFound
        begin
          # Call to current_weather sometimes fails (e.g, 'Boston, MA'), so try current_city instead
          input_array = city.split(",")
          city, state, country = *input_array.map(&:strip)
          country = "US" if country.nil?
          units = "standard"

          # Only standard units are currently supported by current_city and both state and country are required
          data = @client.current_city(city, state, country)
        rescue
          return "Sorry, I couldn't find the weather for #{city}"
        end
      end

      weather = data.main.map { |key, value| "#{key} #{value}" }.join(", ")
      "The current weather in #{data.name} in #{units} units is #{weather}"
    end
  end
end
