defmodule Matrix do

  def new(rows, cols, val \\ 0) do
    for _rows <- 1..rows, do: makeRow(cols,val)
  end

  defp makeRow(n, val) do
    for _row <- 1..n, do: val
  end


  def rand(rows, cols) do
    for _rows <- 1..rows, do: makeRandomRow(cols)
  end

  defp makeRandomRow(n) do
    for _row <- 1..n, do: :random.uniform
  end



  def zeros(rows, cols) do
    new(rows, cols, 0)
  end

  def ones(rows, cols) do
    new(rows, cols, 1)
  end

  def ident(rows) do
    # cols = rows (square matrix)
    x = zeros(rows,rows)
    for r <- Enum.zip(x,0..rows-1), do: List.replace_at( elem(r,0), elem(r,1), 1)
  end



  def size(x) do
    rows = length(x)
    cols = length( List.first(x) )
    {rows, cols}
  end


  def set(x, row, col, val) do
    row_vals = Enum.at(x,row)
    new_row = List.replace_at(row_vals,col,val)
    List.replace_at(x, row, new_row)
  end

  def elem(x, row, col) do
    row_vals = Enum.at(x,row)
    Enum.at(row_vals, col)
  end







  def add(x, y) do
    Stream.zip(x, y) |> Enum.map( fn({a,b})->addRows(a,b) end )
  end

  def addRows(r1, r2) when r1 == []  or  r2 == [], do: []
  def addRows(r1, r2) do
    [h1|t1] = r1
    [h2|t2] = r2
    [h1+h2] ++ addRows(t1,t2)
  end



  def sub(x, y) do
    Stream.zip(x, y) |> Enum.map( fn({a,b})->subRows(a,b) end )
  end

  def subRows(r1, r2) when r1 == []  or  r2 == [], do: []
  def subRows(r1, r2) do
    [h1|t1] = r1
    [h2|t2] = r2
    [h1-h2] ++ subRows(t1,t2)
  end



  def emult(x, y) do
    Stream.zip(x, y) |> Enum.map( fn({a,b})->emultRows(a,b) end )
  end

  def emultRows(r1, r2) when r1 == []  or  r2 == [], do: []
  def emultRows(r1, r2) do
    [h1|t1] = r1
    [h2|t2] = r2
    [h1*h2] ++ emultRows(t1,t2)
  end



end
