# Functions that process each list element in its own process.
defmodule Parallel do

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
