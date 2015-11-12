defmodule Matrix.Mixfile do
  use Mix.Project

  def project do
    [app: :matrix,
     version: "0.3.0",
     description: "",
     elixir: "~> 1.1",
     build_embedded: Mix.env == :dev,
     start_permanent: Mix.env == :dev,
     deps: deps,
     docs: [extras: []]]
  end

  # Configuration for the OTP application
  #
  # Type "mix help compile.app" for more information
  def application do
    [applications: [:logger]]
  end

  # Dependencies can be Hex packages:
  #
  #   {:mydep, "~> 0.3.0"}
  #
  # Or git/path repositories:
  #
  #   {:mydep, git: "https://github.com/elixir-lang/mydep.git", tag: "0.1.0"}
  #
  # Type "mix help deps" for more examples and options
  defp deps do
    [{:earmark, "~> 0.1"},
     {:ex_doc, github: "elixir-lang/ex_doc"},
     {:exprintf, "~> 0.1"}]
  end
end
