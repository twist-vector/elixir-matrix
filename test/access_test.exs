defmodule AccessTest do
  use ExUnit.Case
  doctest Matrix


  test "Check size" do
    #
    # Verify the size function works.  This should return a tuple giving
    # the number of rows and columns in the matrix.  We'll make a random
    # size probably non-square matrix.
    #
    rows = :random.uniform(50)
    cols = :random.uniform(50)

    x = Matrix.new(rows,cols)
    s = Matrix.size(x)

    assert s == {rows, cols}
  end


  test "Element access" do
    #
    # Verify that we can read and modify a particular element in the matrix.
    # We'll make a random size probably non-square matrix and grab a random
    # element to check.
    #
    # fill matrix with -10 since that shouldn't be the same as anything returned
    # by :random and therefore we know we'll be changing something.
    rows = :random.uniform(50)
    cols = :random.uniform(50)
    x = Matrix.new(rows,cols, -10)

    r = :random.uniform(rows-1)
    c = :random.uniform(cols-1)
    v = :random.uniform(50)

    val = Matrix.elem(x,r,c)
    x = Matrix.set(x, r,c, v)
    mod_val = Matrix.elem(x,r,c)

    assert val == mod_val != val
    assert mod_val == v
  end

end
