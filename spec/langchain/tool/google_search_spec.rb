# frozen_string_literal: true

require "google_search_results"

RSpec.describe Langchain::Tool::GoogleSearch do
  subject {
    described_class.new(api_key: "123")
  }

  let(:response) do
    {
      answer_box: {
        type: "organic_result",
        title: "Empire State Building/Height",
        link: "https://www.google.com/search?q=+height&stick=H4sIAAAAAAAAAONgFuLQz9U3MMpLiVeCs7QkQ_JLizKLSxxLSooSk0sy8_OCM1NSyxMrixcxKmYnW-knFiVnZJakJpeUFqXqF5cUlYJZVhmpmekZJYtY2RUgLACBz-fMXwAAAA&sa=X&ved=2ahUKEwiI7pz2h_H-AhXSmYkEHWyoBIcQMSgAegQIOhAB",
        answer: "1,250′, 1,454′ to tip"
      }
    }
  end

  before do
    allow_any_instance_of(GoogleSearch).to receive(:get_hash).and_return(response)
  end

  describe "#execute_search" do
    it "returns the raw hash" do
      expect(subject.execute_search(input: "how tall is empire state building")).to be_a(Hash)
    end
  end

  describe "#execute" do
    it "returns the answer" do
      expect(subject.execute(input: "how tall is empire state building")).to eq("1,250′, 1,454′ to tip")
    end
  end
end
