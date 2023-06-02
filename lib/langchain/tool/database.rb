module Langchain::Tool
  class Database < Base
    description <<~DESC
      Useful for getting the result of a database query.

      The input to this tool should be valid SQL.
    DESC

    # Establish a database connection
    # example: 'postgres://postgres:postgres@localhost:5432/db_name'
    # Empty string creates memory database using sqlite3
    def initialize(db_connection_string = "")
      depends_on "sequel"
      depends_on "sqlite3"
      require "sequel"
      require "sequel/extensions/schema_dumper"

      @db = if db_connection_string.empty?
        Sequel.sqlite
      else
        Sequel.connect(db_connection_string)
      end
      @db.extension :schema_dumper
    end

    def schema
      Langchain.logger.info("Database: Getting schema")
      @db.dump_schema_migration(same_db: true, foreign_keys: false, indexes: false)
    end

    # Evaluates a sql expression
    # @param input [String] sql expression
    # @return [String] results
    def execute(input:)
      Langchain.logger.info("Database: Using the Database Tool with \"#{input}\"")
      results = ""
      @db[input].each do |row|
        results += row.map { |key, value| key.to_s + ": " + value.to_s }.each { |col| p col.to_s }.join(", ")
      end
      results
    end
  end
end
