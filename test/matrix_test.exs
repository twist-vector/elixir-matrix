defmodule MatrixTest do
  use ExUnit.Case

  # Tests currently embedded in documentation
  doctest Matrix  

  test "reshape vector into matrix" do
    v = Enum.map(1..6, &Function.identity/1)
    m = Matrix.reshape(v, {2, 3})

    assert m == [[1, 2], [3, 4], [5, 6]]
  end

  test "reshape matrix into vector" do
    v = Enum.map(1..6, &Function.identity/1)
    m = [[1, 2], [3, 4], [5, 6]]

    assert Matrix.reshape(m, 6) == v
  end

end
