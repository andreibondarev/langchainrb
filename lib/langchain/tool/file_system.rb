# frozen_string_literal: true

module Langchain::Tool
  class FileSystem < Base
    #
    # A tool that wraps the Ruby file system classes.
    #
    # Usage:
    #    file_system = Langchain::Tool::FileSystem.new
    #
    NAME = "file_system"
    FUNCTIONS = [:list_directory, :read_file, :write_to_file]

    def list_directory(directory_path:)
      Dir.entries(directory_path)
    rescue Errno::ENOENT
      "No such directory: #{directory_path}"
    end

    def read_file(file_path:)
      File.read(file_path)
    rescue Errno::ENOENT
      "No such file: #{file_path}"
    end

    def write_to_file(file_path:, content:)
      File.write(file_path, content)
    rescue Errno::EACCES
      "Permission denied: #{file_path}"
    end
  end
end
