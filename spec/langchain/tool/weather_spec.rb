# frozen_string_literal: true

require "open-weather-ruby-client"

RSpec.describe Langchain::Tool::Weather do
  subject {
    described_class.new(api_key: "123")
  }

  let(:response) {
    OpenWeather::Models::City::Weather.new(
      name: "Boston",
      dt: Time.now,
      main: {
        feels_like: 277.73,
        humidity: 81,
        pressure: 1005,
        temp: 282.57,
        temp_max: 283.15,
        temp_min: 281.48
      }
    )
  }

  let(:response_farenheit) {
    OpenWeather::Models::City::Weather.new(
      name: "Chicago",
      dt: Time.now,
      main: {
        feels_like: 85.12,
        humidity: 81,
        pressure: 1005,
        temp: 88.56,
        temp_max: 86.87,
        temp_min: 87.07
      }
    )
  }

  let(:response_on_error) {
    OpenWeather::Models::City::Weather.new(
      timezone: -14400,
      id: 4930956,
      name: "Atlanta",
      cod: 200,
      main: {
        temp: 287.51,
        feels_like: 287.4,
        temp_min: 286.25,
        temp_max: 288.4,
        pressure: 1004,
        humidity: 92
      }
    )
  }

  before do
    allow_any_instance_of(OpenWeather::Client).to receive(:current_weather)
      .with(city: "Boston", units: "standard")
      .and_return(response)

    allow_any_instance_of(OpenWeather::Client).to receive(:current_weather)
      .with(city: "Chicago, IL", units: "imperial")
      .and_return(response_farenheit)

    allow_any_instance_of(OpenWeather::Client).to receive(:current_weather)
      .with(city: "Atlanta, GA", units: "standard")
      .and_raise(Faraday::ResourceNotFound, "404 Not Found")

    allow_any_instance_of(OpenWeather::Client).to receive(:current_city)
      .with("Atlanta", "GA", "US")
      .and_return(response_on_error)
  end

  describe "#execute" do
    it "returns current weather" do
      expect(subject.execute(input: "Boston; standard")).to include("282.57")
    end

    it "returns current weather with different units" do
      expect(subject.execute(input: "Chicago, IL; imperial")).to include("88.56")
    end

    it "returns answer even after an exception" do
      expect(subject.execute(input: "Atlanta, GA; standard")).to include("287.51")
    end
  end
end
