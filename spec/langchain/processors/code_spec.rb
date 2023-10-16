# frozen_string_literal: true

RSpec.describe Langchain::Processors::Code do
  describe "#parse" do
    let(:file) { File.open("spec/fixtures/loaders/foo.rb") }
    let(:text) do
      <<~RUBY
        class Foo
          def bar
            puts "Hello world!"
          end
        end
      RUBY
    end

    it "parses the file and returns the text" do
      expect(described_class.new.parse(file)).to include(text)
    end
  end
end
