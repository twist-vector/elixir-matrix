# Matrix

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

## Installation

Add *matrix* as a dependency in your `mix.exs` file.

```elixir
def deps do
  [ { :matrix, "~> 0.3.0" } ]
end
```

After you are done, run `mix deps.get` in your shell to fetch and compile
Matrix. Start an interactive Elixir shell with `iex -S mix` and try the examples
in the [examples section](#examples).


## Documentation

Documentation for the package is available online via Hex at
[http://hexdocs.pm/matrix](http://hexdocs.pm/matrix).  You can also generate
local docs via the mix task
```elixir
mix docs
```
This will generate the HTML documentation and place it into the `doc` subdirectory.

## [Examples](Examples)
```elixir
iex> Matrix.new(3, 4)
[[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]

iex> Matrix.ident(4)
[[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
```


## License

   Copyright 2015 Thomas Krauss

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

[http://www.apache.org/licenses/LICENSE-2.0](http://www.apache.org/licenses/LICENSE-2.0)

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
