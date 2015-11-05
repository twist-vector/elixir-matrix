defmodule TimingTest do
  use ExUnit.Case
  doctest Matrix


  test "Timings" do
    rows = cols = 2000

    x = Matrix.rand(rows,cols)

    {time1, _res} = :timer.tc( fn -> Matrix.add(x,x) end, [])
    IO.puts "Addition of #{rows}x#{cols}: #{time1/1000} (ms)\n"

    #assert true
  end


end
