defmodule Matrix do
  @moduledoc """
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
  """
  @vsn 1
  @typedoc """
      A list of values representing a matrix row.
  """
  @type row :: [number]
  @type matrix :: [row]

  @comparison_epsilon 1.0e-12
  @comparison_max_ulp 1


  @doc """
    Returns a new matrix of the specified size (number of rows and columns).
    All elements of the matrix are filled with the supplied value "val"
    (default 0).

    #### See also
    [ones/2](#ones/2), [rand/2](#rand/2), [zeros/2](#zeros/2)

    #### Examples
        iex> Matrix.new(3, 4)
        [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]

        iex> Matrix.new(2, 3, -10)
        [[-10, -10, -10], [-10, -10, -10]]
    """
  @spec new(integer, integer, number) :: matrix
  def new(rows, cols, val \\ 0) do
    for _r <- 1..rows, do: make_row(cols,val)
  end

  def make_row(0, _val), do: []
  def make_row(n, val), do: [val] ++ make_row(n-1, val)


  @doc """
    Returns a new matrix of the specified size (number of rows and columns)
    whose elements are sequential starting at 1 and increasing across the row.

    #### See also
    [new/3](#new/3), [ones/2](#ones/2), [rand/2](#rand/2), [zeros/2](#zeros/2)

    #### Examples
        iex> Matrix.seq(3,2)
        [[1, 2], [3, 4], [5, 6]]
    """
  @spec seq(integer, integer) :: matrix
  def seq(rows, cols) do
    for r <- 1..rows, do:
        for c <- 1..cols, do: (r-1)*cols + c
  end



  @doc """
    Returns a new matrix of the specified size (number of rows and columns).
    All elements of the matrix are filled with uniformly distributed random
    numbers between 0 and 1.

    #### Examples
        iex> _ = :rand.seed(:exs1024, {123, 123534, 345345})
        iex> Matrix.rand(3,3)
        [[0.5820506340260994, 0.6739535732076178, 0.9178030245386003],
         [0.7402049520743949, 0.5589108995145826, 0.8687305849540213],
         [0.8851580858928109, 0.988438251464987, 0.18105169154176423]]

    #### See also
    [new/3](#new/3), [ones/2](#ones/2), [zeros/2](#zeros/2)
    """
  @spec rand(integer, integer) :: matrix
  def rand(rows, cols) do
    for _r <- 1..rows, do: make_random_row(cols)
  end

  def make_random_row(0), do: []
  def make_random_row(n), do: [:rand.uniform] ++ make_random_row(n-1)




  @doc """
    Returns a new matrix of the specified size (number of rows and columns).
    All elements of the matrix are filled with the zeros.

    #### See also
    [new/3](#new/3), [ones/2](#ones/2), [rand/2](#rand/2)

    #### Examples
        iex> Matrix.zeros(3, 4)
        [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
    """
  @spec zeros(integer, integer) :: matrix
  def zeros(rows, cols), do: new(rows, cols, 0)


  @doc """
    Returns a new matrix of the specified size (number of rows and columns).
    All elements of the matrix are filled with the ones.

    #### See also
    [new/3](#new/3), [rand/2](#rand/2), [zeros/2](#zeros/2)

    #### Examples
        iex> Matrix.ones(3, 4)
        [[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]]
    """
  @spec ones(integer, integer) :: matrix
  def ones(rows, cols), do: new(rows, cols, 1)


  @doc """
    Returns a new square "diagonal" matrix whose elements are zero except for
    the diagonal.  The diagonal elements will be composed of the supplied list

    #### See also
    [new/3](#new/3), [ones/2](#ones/2), [ident/1](#ident/1)

    #### Examples
        iex> Matrix.diag([1,2,3])
        [[1, 0, 0], [0, 2, 0], [0, 0, 3]]
    """
  @spec diag([number]) :: matrix
  def diag(d) do
    rows = length(d)
    Enum.zip( d, 0..rows-1 )
    |> Enum.map(fn({v,s})->
                  row = [v]++Matrix.make_row(rows-1,0)
                  rrotate(row, s)
                end)
  end
  def lrotate(list, 0), do: list
  def lrotate([head|list], number), do: lrotate(list ++ [head], number - 1)
  def rrotate(list, number), do:
    list
    |> Enum.reverse
    |> lrotate(number)
    |> Enum.reverse


  @doc """
    Returns a new "identity" matrix of the specified size.  The identity is
    defined as a square matrix with ones on the diagonal and zeros in all
    off-diagonal elements.  Since the matrix is square only a single size
    parameter is required.

    #### See also
    [diag/1](#diag/1), [ones/2](#ones/2), [rand/2](#rand/2)

    #### Examples
        iex> Matrix.ident(3)
        [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    """
  @spec ident(integer) :: matrix
  def ident(rows), do: diag(make_row(rows,1))



  @doc """
    Returns the size (dimensions) of the supplied matrix.  The return value is a
    tuple of the dimensions of the matrix as {rows,cols}.

    #### See also
    [new/3](#new/3), [ones/2](#ones/2), [rand/2](#rand/2)

    #### Examples
        iex> Matrix.size( Matrix.new(3,4) )
        {3, 4}
    """
  @spec size(matrix) :: {integer,integer}
  def size(x) do
    rows = length(x)
    cols = length( List.first(x) )
    {rows, cols}
  end


  @doc """
    Returns a matrix that is a copy of the supplied matrix (x) with the
    specified element (row and column) set to the specified value (val).  The
    row and column indices are zero-based.  Negative indices indicate an offset
    from the end of the row or column. If an index is out of bounds, the
    original matrix is returned.

    #### See also
    [elem/3](#elem/3)

    #### Examples
        iex> Matrix.set( Matrix.ident(3), 0,0, -1)
        [[-1, 0, 0], [0, 1, 0], [0, 0, 1]]
    """
  @spec set(matrix, integer, integer, number) :: matrix
  def set(x, row, col, val) do
    row_vals = Enum.at(x,row)
    new_row = List.replace_at(row_vals,col,val)
    List.replace_at(x, row, new_row)
  end


  @doc """
    Returns the value of the specified element (row and column) of the given
    matrix (x).  The row and column indices are zero-based.  Returns `default`
    if either row or col are out of bounds.

    #### See also
    [set/4](#set/4)

    #### Examples
        iex> Matrix.elem( Matrix.ident(3), 0,0 )
        1
    """
  @spec elem(matrix, integer, integer) :: number
  def elem(x, row, col, default \\ nil) do
    row_vals = Enum.at(x,row,nil)
    if row_vals == nil, do: default, else: Enum.at(row_vals, col, default)
  end



  @doc """
    Returns a new matrix whose elements are the sum of the elements of
    the provided matrices.  If the matrices are of differing sizes, the
    returned matrix will be the size and dimensions of the "overlap" between
    them.  For instance, the sum of a 3x3 matrix with a 2x2 matrix will be
    2x2.  The sum of a 3x1 matrix with a 1x3 matrix will be 1x1.

    #### See also
    [sub/2](#sub/2), [emult/2](#emult/2)

    #### Examples
        iex> Matrix.add( Matrix.ident(3), Matrix.ident(3) )
        [[2, 0, 0], [0, 2, 0], [0, 0, 2]]

        iex> Matrix.add( Matrix.ones(3,3), Matrix.ones(2,2) )
        [[2, 2], [2, 2]]

        iex> Matrix.add( Matrix.ones(3,1), Matrix.ones(1,3) )
        [[2]]
    """
  @spec add(matrix, matrix) :: matrix
  def add(x, y) do
    Enum.zip(x, y) |> Enum.map( fn({a,b})->add_rows(a,b) end )
  end




  @doc """
    Returns a new matrix whose elements are the difference (subtraction) of the
    elements of the provided matrices.  If the matrices are of differing sizes,
    the returned matrix will be the size and dimensions of the "overlap" between
    them.  For instance, the difference of a 3x3 matrix with a 2x2 matrix will
    be 2x2. The difference of a 3x1 matrix with a 1x3 matrix will be 1x1.

    #### See also
    [add/2](#add/2), [emult/2](#emult/2)

    #### Examples
        iex> Matrix.sub( Matrix.ident(3), Matrix.ones(3,3) )
        [[0, -1, -1], [-1, 0, -1], [-1, -1, 0]]

        iex> Matrix.sub( Matrix.ones(3,3), Matrix.ones(2,2) )
        [[0, 0], [0, 0]]

        iex> Matrix.sub( Matrix.ones(3,1), Matrix.ones(1,3) )
        [[0]]
    """
  @spec sub(matrix, matrix) :: matrix
  def sub(x, y) do
    Enum.zip(x, y) |> Enum.map( fn({a,b})->subtract_rows(a,b) end )
  end




  @doc """
    Returns a new matrix whose elements are the element-by-element multiply of
    the elements of the provided matrices.  Note that this is not the linear
    algebra matrix multiply.  If the matrices are of differing sizes, the
    returned matrix will be the size and dimensions of the "overlap" between
    them.  For instance, the element multiply of a 3x3 matrix with a 2x2 matrix
    will be 2x2, for a 3x1 matrix with a 1x3 matrix will be 1x1.

    #### See also
    [add/2](#add/2), [sub/2](#sub/2)

    #### Examples
        iex> Matrix.emult( Matrix.new(3,3,2), Matrix.new(3,3,-2) )
        [[-4, -4, -4], [-4, -4, -4], [-4, -4, -4]]

        iex> Matrix.emult( Matrix.ones(3,3), Matrix.ones(2,2) )
        [[1, 1], [1, 1]]

        iex> Matrix.emult( Matrix.ones(3,1), Matrix.ones(1,3) )
        [[1]]
    """
  @spec emult(matrix, matrix) :: matrix
  def emult(x, y) do
    Enum.zip(x, y) |> Enum.map( fn({a,b})->emult_rows(a,b) end )
  end



  @doc """
    Returns a new matrix which is the linear algebra matrix multiply
    of the provided matrices.  It is required that the number of columns
    of the first matrix (x) be equal to the number of rows of the second
    matrix (y).  If x is an NxM and y is an MxP, the returned matrix product
    xy is NxP.  If the number of columns of x does not equal the number of
    rows of y an `ArgumentError` exception is thrown with the message "sizes
    incompatible"

    #### See also
    [emult/2](#emult/2)

    #### Examples
        iex> Matrix.mult( Matrix.seq(2,2), Matrix.seq(2,2) )
        [[7, 10], [15, 22]]

        iex> Matrix.mult( Matrix.ones(3,2), Matrix.ones(2,3) )
        [[2, 2, 2], [2, 2, 2], [2, 2, 2]]

        iex> Matrix.mult( Matrix.ones(3,2), Matrix.ones(3,2) )
        ** (ArgumentError) sizes incompatible
    """
  @spec mult(matrix, matrix) :: matrix
  def mult(x, y) do

    {_rx,cx} = size(x)
    {ry,_cy} = size(y)
    if (cx != ry), do:
      raise ArgumentError, message: "sizes incompatible"

    trans_y = transpose(y)
    Enum.map(x, fn(row)->
                      Enum.map(trans_y, &dot_product(row, &1))
                    end)
  end

  defp dot_product(r1, _r2) when r1 == [], do: 0
  defp dot_product(r1, r2) do
    [h1|t1] = r1
    [h2|t2] = r2
    (h1*h2) + dot_product(t1, t2)
  end



  @doc """
    Returns a new matrix whose elements are the transpose of the supplied matrix.
    The transpose essentially swaps rows for columns - that is, the first row
    becomes the first column, the second row becomes the second column, etc.

      #### Examples
      iex> Matrix.transpose( Matrix.seq(3,2) )
      [[1, 3, 5], [2, 4, 6]]
  """
  @spec transpose(matrix) :: matrix
  def transpose(m) do
    swap_rows_cols(m)
  end

  defp swap_rows_cols( [h|_t] ) when h==[], do: []
  defp swap_rows_cols(rows) do
    firsts = Enum.map(rows, fn(x) -> hd(x) end) # first element of each row
    rest = Enum.map(rows, fn(x) -> tl(x) end)   # remaining elements of each row
    [firsts | swap_rows_cols(rest)]
  end




  @doc """
  Returns a new matrix which is the (linear algebra) inverse of the supplied
  matrix.  If the supplied matrix is "x" then, by definition,
          x * inv(x) = I
  where I is the identity matrix.  This function uses a brute force Gaussian
  elimination so it is not expected to be terribly fast.

  #### Examples
      iex> x = Matrix.rand(5,5)
      iex> res = Matrix.mult( x, Matrix.inv(x) )
      iex> Matrix.almost_equal(res,[[1,0,0],[0,1,0],[0,0,1]])
      true
  """
  @spec inv(matrix) :: matrix
  def inv(x) do
    {rows,_cols} = size(x)
    y = row_reduce( supplement(x,ident(rows)) )

    {yl,yr} = desupplement(y)
    z = supplement( full_flip(yl), full_flip(yr) )
    {_,zzr} = desupplement( row_reduce(z) )
    full_flip(zzr)
  end



  @doc """
  Returns a new matrix whose elements are identical to the supplied matrix x
  but with the supplied value appended to the beginning of each row.

  #### See also
  [postfix_row/2](#postfix_rows/2)

  #### Examples
      iex> Matrix.prefix_rows( Matrix.seq(2,2), 10 )
      [[10, 1, 2], [10, 3, 4]]
  """
  @spec prefix_rows(matrix, number) :: matrix
  def prefix_rows(x, val) do
    Enum.map(x, fn(r) -> [val]++r end)
  end


  @doc """
  Returns a new matrix whose elements are identical to the supplied matrix x
  but with the supplied value appended to the end of each row.

  #### See also
  [prefix_rows/2](#prefix_rows/2)

  #### Examples
      iex> Matrix.postfix_rows( Matrix.seq(2,2), 10 )
      [[1, 2, 10], [3, 4, 10]]
  """
  @spec postfix_rows(matrix, number) :: matrix
  def postfix_rows(x, val) do
    Enum.map(x, fn(r) -> r++[val] end)
  end


  @doc """
  Compares two matrices as being (approximately) equal.  Since floating point
  numbers have slightly different representations and accuracies on different
  architectures it is generally not a good idea to compare them directly.
  Rather numbers are considered equal if they are within an "epsilon" of each
  other.  *almost_equal* compares all elements of two matrices, returning true
  if all elements are within the provided epsilon.

  #### Examples
      iex> Matrix.almost_equal( [[1, 0], [0, 1]], [[1,0], [0,1+1.0e-12]] )
      false

      iex> Matrix.almost_equal( [[1, 0], [0, 1]], [[1,0], [0,1+0.5e-12]] )
      true
  """
  @spec almost_equal(matrix, matrix, number, number) :: boolean
  def almost_equal(x, y, eps \\ @comparison_epsilon, max_ulp \\ @comparison_max_ulp) do
    Enum.zip(x,y)
    |> Enum.map(fn({r1,r2})->rows_almost_equal(r1, r2, eps, max_ulp) end)
    |> Enum.all?
  end


  @doc """
  Returns a new matrix whose elements are the elements of matrix x multiplied by
  the scale factor "s".

  #### Examples
      iex> Matrix.scale( Matrix.ident(3), 2 )
      [[2,0,0], [0,2,0], [0,0,2]]

      iex> Matrix.scale( Matrix.ones(3,4), -2 )
      [[-2, -2, -2, -2], [-2, -2, -2, -2], [-2, -2, -2, -2]]
  """
  @spec scale(matrix, number) :: matrix
  def scale(x, s) do
    Enum.map(x, fn(r)->scale_row(r,s) end)
  end

  @doc """
  Returns a new matrix where each element is the result of invoking `f` on each
  element of the matrix `x`

  #### Examples
      iex> Matrix.map(Matrix.zeros(3, 3), fn x -> x + 4 end)
      [[4,4,4], [4,4,4], [4,4,4]]
  """
  def map(x, f) do
    Enum.map(x, fn r -> map_row(r, f) end)
  end

  @doc """
  Reshapes vector into matrix and matrix into vector

  #### Examples
      iex> Matrix.reshape([1,2,3,4,5,6], {2, 3})
      [[1,2], [3,4], [5,6]]

      iex> Matrix.reshape([[1,2], [3,4], [5,6]], 6)
      [1,2,3,4,5,6]
  """
  def reshape(vector, {row, column}) do
    if length(vector) != row * column do
      raise ArgumentError, message: "incompatible shape"
    end

    Enum.chunk_every(vector, row)
  end

  def reshape(m, length) do
    vector = List.flatten(m)

    if length(vector) != length do
      raise ArgumentError, message: "incompatible shape"
    end

    vector
  end

  @doc """
  Returns a string which is a "pretty" representation of the supplied
  matrix.
  """
  @spec pretty_print(matrix, charlist, charlist) :: atom
  def pretty_print(m, fmt\\"%d", sep\\"") do
    str = m
          |> Enum.map(fn(r)->show_row(r,fmt,sep) <> "\n" end)
          |> Enum.join("")
    IO.puts(str)
  end


  @doc """
  Returns the Kronecker tensor product of two matrices A and B. If A is an
  MxN and B is PxQ, then the returned matrix is an (M*P)x(N*Q) matrix formed
  by taking all possible products between the elements of A and the matrix B.

  <pre>A = |1000|      B = |  1 -1|
      |0100|          | -1  1|
      |0010|
      |0001|                  </pre>

  then <pre>     kron(A,B) = |  1 -1  0  0  0  0  0  0|
                   | -1  1  0  0  0  0  0  0|
                   |  0  0  1 -1  0  0  0  0|
                   |  0  0 -1  1  0  0  0  0|
                   |  0  0  0  0  1 -1  0  0|
                   |  0  0  0  0 -1  1  0  0|
                   |  0  0  0  0  0  0  1 -1|
                   |  0  0  0  0  0  0 -1  1|</pre>
  """
  @spec kron(matrix, matrix) :: matrix
  def kron([], _b), do: []
  def kron([h|t], b) do
    row_embed(h, b, [])++kron(t, b)
  end



  #################
  # Private supporting functions
  #################



  #
  # These functions apply a specific math operation to all the elements of the
  # supplied rows.  They are used by the math routine (e.g., add).  They call the
  # recursive element adding functions to continually append new elements to
  # the row.  Note that this was found to be faster than a simple list
  # comprehension or Enum.map, at least using Erlang/OTP 18 and Elixir 1.1.1
  #
  defp add_rows(r1, r2) when r1 == []  or  r2 == [], do: []
  defp add_rows(r1, r2) do
    [h1|t1] = r1
    [h2|t2] = r2
    [h1+h2] ++ add_rows(t1,t2)
  end

  defp subtract_rows(r1, r2) when r1 == []  or  r2 == [], do: []
  defp subtract_rows(r1, r2) do
    [h1|t1] = r1
    [h2|t2] = r2
    [h1-h2] ++ subtract_rows(t1,t2)
  end

  defp emult_rows(r1, r2) when r1 == []  or  r2 == [], do: []
  defp emult_rows(r1, r2) do
    [h1|t1] = r1
    [h2|t2] = r2
    [h1*h2] ++ emult_rows(t1,t2)
  end



  #
  # Support function for matrix inverse.  "Supplement" the supplied matrix (x)
  # by appending the matrix y to the right of it.  Generally used to make,
  # for example,
  #      x = |1 2|
  #          |3 4|
  # supplemented with the 2x2 identity matrix becomes
  #      x = |1 2 1 0|
  #          |3 4 0 1|
  #
  defp supplement([],y), do: y
  defp supplement(x,y) do
    Enum.zip(x,y)
    |> Enum.map(fn({r1,r2}) -> r1++r2 end)
  end

  #
  # Inverse of the "supplement" function.  This breaks the matrix apart into
  # a left and right part - returned as a tuple.  For example for
  #      x = |1 2 1 0|
  #          |3 4 0 1|
  # desupplement returns
  #      { |1 2|   |1 0|
  #        |3 4|   |0 1| }
  #
  defp desupplement(x) do
    {_rows,cols} = size(x)
    left = Enum.map(x, fn(r) -> elem(Enum.split(r,round(cols/2)),0) end)
    right = Enum.map(x, fn(r) -> elem(Enum.split(r,round(cols/2)),1) end)
    {left,right}
  end

  #
  # Multiplies a row by a (scalar) constant.
  #
  defp scale_row(r, v) do
    map_row(r, fn(x) -> x * v end)
  end

  #
  # Maps over a row
  #
  defp map_row(r, f) do
    Enum.map(r, f)
  end

  #
  # Uses elementary row operations to reduce the supplied matrix to row echelon
  # form.  This is the first step of matrix inversion using Gaussian elimination.
  #
  defp row_reduce([]), do: []
  defp row_reduce(rows) do
    firsts = Enum.map(rows, fn(x) -> hd(x) end) # first element of each row
    s = hd(firsts)
    y = if abs(s)<1.0e-10 do
          scale_row(hd(rows), 1)
        else
          scale_row(hd(rows), 1/s)
        end

    first_rest = Enum.map(tl(rows), fn(x) -> hd(x) end)
    z = Enum.zip(tl(rows),first_rest)
        |> Enum.map(fn({r,v}) ->
                      if abs(v)<1.0e-10 do
                        tl(r)
                      else
                        tl(subtract_rows(scale_row(r,1/v),y))
                      end
                    end)

    [y] ++ prefix_rows(row_reduce(z), 0)
  end

  #
  # Used in the "inv" function.  This function flips the supplied matrix top
  # to bottom and left to right.  It is used after the first reduction to
  # reduced echelon form to allow for recursive back substitution to get the
  # inverse.
  #
  defp full_flip(x) do
    Enum.reverse( Enum.map( x, fn(r)->Enum.reverse(r) end ) )
  end




  #
  # Embed the supplied matrix into a "row" matrix.  Each instance of the embedded
  # matrix is scaled by the element of r
  #
  defp row_embed([], _b, macc), do: macc
  defp row_embed(r, b, macc) do
    s = hd(r)
    row_embed( tl(r), b, supplement(macc,scale(b,s)) )
  end


  #
  # Pretty prints the values in a supplied row.
  #
  defp show_row(r, fmt, sep) do
    str = r
    |> Enum.map(fn(e)->ExPrintf.sprintf(fmt, [e]) end)
    |> Enum.join(sep)
    "|" <> str <> "|"
  end



  #
  # The following functions are used for floating point comparison of matrices.
  #
  # Compares two rows as being (approximately) equal.
  defp rows_almost_equal(r1, r2, eps, max_ulp) do
    x = Enum.zip(r1,r2)
        |> Enum.map(fn({x,y})->close_enough?(x, y, eps, max_ulp) end)
    Enum.all?(x)
  end

  @doc """
  Code borrowed from the ExMath library and duplicated here to reduce
  dependencies.  ExMath is copyright © 2015 Ookami Kenrou <ookamikenrou@gmail.com>

  Equality comparison for floating point numbers, based on
  [this blog post](https://randomascii.wordpress.com/2012/02/25/comparing-floating-point-numbers-2012-edition/)
  by Bruce Dawson.
  """
  @spec close_enough?(number, number, number, non_neg_integer) :: boolean
  def close_enough?(a, b, epsilon, max_ulps) do
    a = :erlang.float a
    b = :erlang.float b

    cond do
      abs(a - b) <= epsilon -> true

      signbit(a) != signbit(b) -> false

      ulp_diff(a, b) <= max_ulps -> true

      true -> false
    end
  end

  @spec signbit(float) :: boolean
  defp signbit(x) do
    case <<x :: float>> do
      <<1 :: 1, _ :: bitstring>> -> true
      _ -> false
    end
  end

  @spec ulp_diff(float, float) :: integer
  def ulp_diff(a, b), do: abs(as_int(a) - as_int(b))

  @spec as_int(float) :: non_neg_integer
  defp as_int(x) do
    <<int :: 64>> = <<x :: float>>
    int
  end


end
