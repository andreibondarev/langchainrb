# frozen_string_literal: true

module Langchain
  module Processors
    class Xlsx < Base
      EXTENSIONS = [".xlsx", ".xlsm"].freeze
      CONTENT_TYPES = ["application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"].freeze

      def initialize(*)
        depends_on "roo"
        require "roo"
      end

      # Parse the document and return the text
      # @param [File] data
      # @return [String]
      def parse(data)
        xlsx_file = Roo::Spreadsheet.open(data)
        xlsx_file.each_with_pagename.flat_map do |_, sheet|
          sheet.map do |row|
            row.map { |i| i.to_s.strip }
          end
        end
      end
    end
  end
end
