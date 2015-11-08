defmodule Matrix do
  @moduledoc """
  *Matrix* is a linear algebra library for manipulating dense matrices. Its
  design goals are to be reasonably efficient in terms of both memory usage and
  computations.  Unfortunately there is a trade off between memory efficiency
  and computational efficiency.  Where these requirements conflict *Matrix*
  will use the more computationally efficient algorithm.

  A secondary design consideration is for ease of use.  It is desirable that the
  *Matrix* package interact with standard Elixir language constructs and other
  packages.  The underlying storage mechanism is, therefore, Elixir lists.

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
  @doc """
      A list of values representing a matrix row.
  """
  @type row :: [number]
  @type matrix :: [row]


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
        iex> _ = :random.seed(12345)
        iex> Matrix.rand(3,3)
        [[0.07797290969719865, 0.3944785128151924, 0.9781224924937147],
         [1.3985610037403617e-4, 0.5536761216397539, 0.35476183770551284],
         [0.7021763747372531, 0.5537966721193639, 0.1607491687700906]]

    #### See also
    [new/3](#new/3), [ones/2](#ones/2), [zeros/2](#zeros/2)
    """
  @spec rand(integer, integer) :: matrix
  def rand(rows, cols) do
    for _r <- 1..rows, do: make_random_row(cols)
  end



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
  def zeros(rows, cols) do
    new(rows, cols, 0)
  end


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
  def ones(rows, cols) do
    new(rows, cols, 1)
  end


  @doc """
    Returns a new "identity" matrix of the specified size.  The identity is
    defined as a square matrix with ones on the diagonal and zeros in all
    off-diagonal elements.  Since the matrix is square only a single size
    parameter is required.

    #### See also
    [new/3](#new/3), [ones/2](#ones/2), [rand/2](#rand/2)

    #### Examples
        iex> Matrix.ident(3)
        [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    """
  @spec ident(integer) :: matrix
  def ident(rows) do
    # cols = rows (square matrix)
    x = zeros(rows,rows)
    for r <- Enum.zip(x,0..rows-1), do: List.replace_at( elem(r,0), elem(r,1), 1)
  end


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
    Returns a matrix that is a copy of the supplied matrix (x) with the specified
    element (row and column) set to the specified value (val).  The row and
    column indices are zero-based.

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
    Returns a value of the specified element  (row and column) of the given
    matrix (x).  The row and column indices are zero-based.

    #### See also
    [set/4](#set/4)

    #### Examples
        iex> Matrix.elem( Matrix.ident(3), 0,0 )
        1
    """
  @spec elem(matrix, integer, integer) :: number
  def elem(x, row, col) do
    row_vals = Enum.at(x,row)
    Enum.at(row_vals, col)
  end



  @doc """
    Returns a new matrix whose elements are the sum of the elements of
    the provided matrices.

    #### See also
    [sub/2](#sub/2), [emult/2](#emult/2)

    #### Examples
        iex> Matrix.add( Matrix.ident(3), Matrix.ident(3) )
        [[2, 0, 0], [0, 2, 0], [0, 0, 2]]
    """
  @spec add(matrix, matrix) :: matrix
  def add(x, y) do
    Enum.zip(x, y) |> Enum.map( fn({a,b})->add_rows(a,b) end )
  end




  @doc """
    Returns a new matrix whose elements are the difference (subtraction) of
    the elements of the provided matrices.

    #### See also
    [add/2](#add/2), [emult/2](#emult/2)

    #### Examples
        iex> Matrix.sub( Matrix.ident(3), Matrix.ones(3,3) )
        [[0, -1, -1], [-1, 0, -1], [-1, -1, 0]]
    """
  @spec sub(matrix, matrix) :: matrix
  def sub(x, y) do
    Enum.zip(x, y) |> Enum.map( fn({a,b})->subtract_rows(a,b) end )
  end




  @doc """
    Returns a new matrix whose elements are the element-by-element multiply
    of the elements of the provided matrices.  Note that this is not the
    linear algebra matrix multiply.

    #### See also
    [add/2](#add/2), [sub/2](#sub/2)

    #### Examples
        iex> Matrix.emult( Matrix.new(3,3,2), Matrix.new(3,3,-2) )
        [[-4, -4, -4], [-4, -4, -4], [-4, -4, -4]]
    """
  @spec emult(matrix, matrix) :: matrix
  def emult(x, y) do
    Enum.zip(x, y) |> Enum.map( fn({a,b})->emult_rows(a,b) end )
  end

  @doc """
    Returns a new matrix which is the linear algebra matrix multiply
    of the provided matrices.

    #### See also
    [emult/2](#emult/2)

    #### Examples
        iex> Matrix.mult( Matrix.seq(2,2), Matrix.seq(2,2) )
        [[7, 10], [15, 22]]

        iex> _ = :random.seed(12345)
        iex> Matrix.mult( Matrix.rand(3,3), Matrix.rand(3,3) )
        [[1.0381005406028176, 0.7212441594338046, 0.9307137121837732],
         [0.6255442219186653, 0.4956710926346579, 0.574699071565491],
         [0.66479248303266, 0.7764927657589071, 1.0048634412082678]]
    """
  @spec mult(matrix, matrix) :: matrix
  def mult(x, y) do
    trans_y = transpose(y)
    Enum.map(x, fn(row)->
                  Enum.map(trans_y, &dot_product(row, &1))
                end)
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
  def transpose(cells) do
    swap_rows_cols(cells)
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
      iex> Matrix.almost_equal(res,[[1,0,0],[0,1,0],[0,0,1]],1.0e-10)
      true
    """
  @spec inv(matrix) :: matrix
  def inv(x) do
    {rows,_cols} = size(x)
    y = reduce( supplement(x,ident(rows)) )

    {yl,yr} = desupplement(y)
    z = supplement( full_flip(yl), full_flip(yr) )
    {_zzl,zzr} = desupplement( reduce(z) )
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
  def postfix_rows(x, val) do
    Enum.map(x, fn(r) -> r++[val] end)
  end



  #################
  # Private supporting functions
  #################

  #
  # These functions create and return a row populated with various values.  They
  # are used by the creation routine (e.g., new).  They call the recursive
  # element adding functions "add_row_element" to continually append new elements to
  # the row.  Note that this was found to be faster than a simple list
  # comprehension or Enum.map, at least using Erlang/OTP 18 and Elixir 1.1.1
  #
  defp make_row(n, val) do
    # The was found to be slower than the recursive call
    #for _r <- 1..n, do: val
    add_row_element([],n,val)
  end
  defp add_row_element(r,0,_), do: r
  defp add_row_element(r,n,val) do
    [val] ++ add_row_element(r,n-1,val)
  end

  defp make_random_row(n) do
    #for _r <- 1..n, do: :random.uniform
    add_random_row_element([],n)
  end
  defp add_random_row_element(r,0), do: r
  defp add_random_row_element(r,n) do
    [:random.uniform] ++ add_random_row_element(r,n-1)
  end



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
  # Recursive support function for "transpose".  This function separates out the
  # first element of each row and forms it into a new row.  It then calls itself
  # with the remaining elements to form the remaining swapped rows.
  #
  defp swap_rows_cols( [h|_t] ) when h==[], do: []
  defp swap_rows_cols(rows) do
    firsts = Enum.map(rows, fn(x) -> hd(x) end) # first element of each row
    rest = Enum.map(rows, fn(x) -> tl(x) end)   # remaining elements of each row
    [firsts | swap_rows_cols(rest)]
  end

  #
  # Support function for matrix multiply.  Computes the dot product of the
  # supplied lists.  Presumably r1 and r2 are rows of a matrix.
  #
  defp dot_product(r1, r2) do
    Stream.zip(r1, r2)
    |> Enum.map(fn({x, y}) -> x * y end)
    |> Enum.sum
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
    Enum.map(r, fn(x) -> x * v end)
  end

  #
  # Uses elementary row operations to reduce the supplied matrix to row echelon
  # form.  This is the first step of matrix inversion using Gaussian elimination.
  #
  defp reduce([]), do: []
  defp reduce(rows) do
    crows = rows
    firsts = Enum.map(crows, fn(x) -> hd(x) end) # first element of each row
    s = hd(firsts)
    y = if abs(s)<1.0e-10 do
          scale_row(hd(crows), 1)
        else
          scale_row(hd(crows), 1/s)
        end

    first_rest = Enum.map(tl(crows), fn(x) -> hd(x) end)
    z = Enum.zip(tl(crows),first_rest)
        |> Enum.map(fn({r,v}) ->
                      if abs(v)<1.0e-10 do
                        tl(r)
                      else
                        tl(subtract_rows(scale_row(r,1/v),y))
                      end
                    end)

    [y] ++ prefix_rows(reduce(z), 0)
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
  # The following functions are used for floating point comparison of matrices.
  #
  # Compares two matrices as being (approximately) equal.
  def almost_equal(x, y, eps) do
    x = Enum.zip(x,y)
        |> Enum.map(fn({r1,r2})->rows_almost_equal(r1, r2, eps) end)
    Enum.all?(x)
  end
  # Compares two rows as being (approximately) equal.
  def rows_almost_equal(r1, r2, eps) do
    x = Enum.zip(r1,r2)
        |> Enum.map(fn({x,y})->vals_almost_equal(x, y, eps) end)
    Enum.all?(x)
  end
  # Compares two floats as being (approximately) equal.
  def vals_almost_equal(x, y, eps) do
    abs(x-y) < eps
  end

end
