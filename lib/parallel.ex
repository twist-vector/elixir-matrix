# Functions that process each list element in its own process.
defmodule Parallel do

 def each(collection, fun) do 
   me = self
   collection
   |> Enum.map(fn(elem) ->
        spawn_link fn -> (send me, { self, fun.(elem) }) end
      end)
   |> Enum.each(fn(pid) ->
        receive do { ^pid, _ } -> :ok end
      end)
 end
 
 # This code came from the book 'Programming Elixir' by Dave Thomas.
 def map(collection, fun) do
   me = self
   collection
   |> Enum.map(fn(elem) ->
        spawn_link fn -> (send me, { self, fun.(elem) }) end
      end)
   |> Enum.map(fn(pid) ->
        receive do { ^pid, result } -> result end
      end)
 end

end
