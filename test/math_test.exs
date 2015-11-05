defmodule MathTest do
  use ExUnit.Case
  doctest Matrix

  test "Addition" do
    #
    # Verify that the element-by-element addition works.  We'll make a random
    # size probably non-square matrix and add it to itself then we'll check
    # some random elements to verify they've been doubled.
    #

    rows = :random.uniform(100)
    cols = :random.uniform(100)
    x = Matrix.rand(rows,cols)
    y = Matrix.add(x,x)

    # We'll just check a few random locations...
    for _loop <- 1..50 do
      r = :random.uniform(rows-1)
      c = :random.uniform(cols-1)

      old_val = Matrix.elem(x,r,c)
      new_val = Matrix.elem(y,r,c)

      # Check the math part of "add"
      assert new_val == 2*old_val
    end
  end


  test "Subtraction" do
    #
    # Verify that the element-by-element subtraction works.  We'll make two
    # random size probably non-square matricies and subtract one from the other.
    # Then we'll check some random elements to verify they're correct.
    #

    rows = :random.uniform(100)
    cols = :random.uniform(100)
    val1 = :random.uniform
    val2 = :random.uniform
    x = Matrix.new(rows,cols,val1)
    y = Matrix.new(rows,cols,val2)
    d = Matrix.sub(x,y)

    # We'll just check a few random locations...
    for _loop <- 1..50 do
      r = :random.uniform(rows-1)
      c = :random.uniform(cols-1)

      val = Matrix.elem(d,r,c)
      assert val == val1-val2
    end
  end


  test "Element-by-element multiplication" do
    #
    # Verify that the element-by-element multiplication works.  We'll make two
    # random size probably non-square matricies and multiply them together.
    # Then we'll check some random elements to verify they're correct.
    #

    rows = :random.uniform(100)
    cols = :random.uniform(100)
    x = Matrix.rand(rows,cols)
    y = Matrix.rand(rows,cols)
    p = Matrix.emult(x,y)

    # We'll just check a few random locations...
    for _loop <- 1..50 do
      r = :random.uniform(rows-1)
      c = :random.uniform(cols-1)

      val1 = Matrix.elem(x,r,c)
      val2 = Matrix.elem(y,r,c)
      prod_val = Matrix.elem(p,r,c)
      assert prod_val == val1*val2
    end
  end

end
