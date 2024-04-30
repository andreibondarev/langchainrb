require "webmock"
require "vcr"

VCR.configure do |config|
  config.cassette_library_dir = "spec/fixtures/vcr_cassettes"
  config.hook_into :webmock
  config.default_cassette_options = {
    record: :once,
    record_on_error: false,
    match_requests_on: %i[method uri body]
  }

  config.filter_sensitive_data("<API_KEY_PLACEHOLDER>") { ENV.fetch("OPENAI_API_KEY") }
end

RSpec.configure do |config|
  config.around(:each) do |example|
    vcr = example.metadata[:vcr]

    if vcr.nil?
      WebMock.disable!
      VCR.turned_off { example.run }
      WebMock.enable!
    else
      VCR.insert_cassette example.full_description
      example.run
      VCR.eject_cassette example.full_description
    end
  end
end
