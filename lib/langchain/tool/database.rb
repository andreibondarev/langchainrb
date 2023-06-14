module Langchain::Tool
  class Database < Base
    #
    # Connects to a database, executes SQL queries, and outputs DB schema for Agents to use
    #
    # Gem requirements: gem "sequel", "~> 5.68.0"
    #

    NAME = "database"

    description <<~DESC
      Useful for getting the result of a database query.

      The input to this tool should be valid SQL.
    DESC

    attr_reader :db

    #
    # Establish a database connection
    #
    # @param connection_string [String] Database connection info, e.g. 'postgres://user:password@localhost:5432/db_name'
    # @return [Database] Database object
    #
    def initialize(connection_string:)
      depends_on "sequel"
      require "sequel"
      require "sequel/extensions/schema_dumper"

      raise StandardError, "connection_string parameter cannot be blank" if connection_string.empty?

      @db = Sequel.connect(connection_string)
      @db.extension :schema_dumper
    end

    #
    # Returns the database schema
    #
    # @return [String] schema
    #
    def schema
      Langchain.logger.info("Dumping schema", for: self.class)
      db.dump_schema_migration(same_db: true, indexes: false) unless db.adapter_scheme == :mock
    end

    #
    # Evaluates a sql expression
    #
    # @param input [String] sql expression
    # @return [Array] results
    #
    def execute(input:)
      Langchain.logger.info("Executing \"#{input}\"", for: self.class)

      db[input].to_a
    rescue Sequel::DatabaseError => e
      Langchain.logger.error(e.message, for: self.class)
    end
  end
end
