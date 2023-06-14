# frozen_string_literal: true

require "pathname"
require "json"
require "yaml"
require "langchain"
require "pry-byebug"

RUNNING_ON_CI = ENV["CI"] == "true"

RSpec.configure do |config|
  # Enable flags like --only-failures and --next-failure
  config.example_status_persistence_file_path = ".rspec_status"

  # Disable RSpec exposing methods globally on `Module` and `main`
  config.disable_monkey_patching!

  config.expect_with :rspec do |c|
    c.syntax = :expect
  end

  # this allows focusing a spec via `fit`, `fcontext`, `fdescribe`, or focus: true
  # don't allow focusing on CI
  config.filter_run focus: !RUNNING_ON_CI
  config.run_all_when_everything_filtered = true
end
