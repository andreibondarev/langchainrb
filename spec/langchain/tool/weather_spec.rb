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

  before do
    allow(subject.client).to receive(:current_weather)
      .with(city: "Boston", units: "standard")
      .and_return(response)

    allow(subject.client).to receive(:current_weather)
      .with(city: "Chicago", units: "imperial")
      .and_return(response_farenheit)
  end

  describe "#execute" do
    it "returns current weather" do
      expect(subject.execute(input: "Boston; standard")).to include("282.57")
    end

    it "returns current weather with units" do
      expect(subject.execute(input: "Chicago; imperial")).to include("88.56")
    end
  end
end
