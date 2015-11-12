# Matrix

*Matrix* is a linear algebra library for manipulating dense matrices. Its
primary design goal is ease of use.  It is desirable that the *Matrix* package
interact with standard Elixir language constructs and other packages.  The
underlying storage mechanism is, therefore, Elixir lists.

A secondary design consideration is for the module to be reasonably efficient
in terms of both memory usage and computations.  Unfortunately there is a
trade off between memory efficiency and computational efficiency.  Where these
requirements conflict *Matrix* will use the more computationally efficient
algorithm.

Each matrix is represented as a "list of lists" whereby a 3x4 matrix is
represented by a list of three items, with each item a list of 4 values.
Constructors are provided for several common matrix configurations including
zero filled, one filled, random filled the identity matrix, etc.

## Examples

    iex> Matrix.new(3, 4)
    [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]

    iex> Matrix.ident(4)
    [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]

## Installation

If [available in Hex](https://hex.pm/docs/publish), the package can be installed as:

  1. Add matrix to your list of dependencies in `mix.exs`:

        def deps do
          [{:matrix, "~> 0.0.1"}]
        end

  2. Ensure matrix is started before your application:

        def application do
          [applications: [:matrix]]
        end
