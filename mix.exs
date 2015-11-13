defmodule Matrix.Mixfile do
  use Mix.Project

  def project do
    [app: :matrix,
     version: "0.3.0",
     description: description,
     package: package,
     elixir: "~> 1.1",
     deps: deps,
     docs: [extras: []]]
  end

  # Configuration for the OTP application
  #
  # Type "mix help compile.app" for more information
  def application do
    [applications: [:logger]]
  end

  defp deps do
    [{:earmark, "~> 0.1"},
     {:ex_doc, github: "elixir-lang/ex_doc"},
     {:exprintf, "~> 0.1"}]
  end

  defp description do
    """
    Matrix is a linear algebra library for manipulating dense matrices. Its
    primary design goal is ease of use.
    """
  end

  defp package do
    [# These are the default files included in the package
     maintainers: ["Tom Krauss"],
     licenses: ["Apache 2.0"],
     links: %{"GitHub" => "https://github.com/twist-vector/elixir-matrix.git",
              "Docs" => "http://http://hexdocs.pm/matrix"}]
  end

end
