defmodule MatrixTest do
  use ExUnit.Case
  doctest Matrix


  test "Constructors" do
    #
    # Check "new" and non-square sizes
    #
    x = Matrix.new(3,2,3)
    assert x == [[3, 3], [3, 3], [3, 3]]
    # ... non-square the other way
    x = Matrix.new(2,3,3)
    assert x == [[3, 3, 3], [3, 3, 3]]

    #
    # Verify the helper constructors that make specific types
    # of matrices
    #
    x = Matrix.zeros(2,2)
    assert x == [[0, 0], [0, 0]]

    x = Matrix.ones(2,2)
    assert x == [[1, 1], [1, 1]]

    x = Matrix.ident(4)
    assert x == [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
  end


end
