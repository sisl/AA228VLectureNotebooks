### A Pluto.jl notebook ###
# v0.20.4

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    #! format: off
    quote
        local iv = try Base.loaded_modules[Base.PkgId(Base.UUID("6e696c72-6542-2067-7265-42206c756150"), "AbstractPlutoDingetjes")].Bonds.initial_value catch; b -> missing; end
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : iv(el)
        el
    end
    #! format: on
end

# ╔═╡ 8005fa69-78ea-40a8-9cda-8ea33470f899
using PlutoUI # For use of @bind, Slider, and TableOfContents

# ╔═╡ 238efdef-6820-48dd-a24f-9b7c8f108f5d
using AbstractTrees

# ╔═╡ d50862fe-4e9d-4dd8-9278-1f935ace223b
using Printf

# ╔═╡ 1488d038-e6ea-4b6d-951f-c424b3983d11
using StanfordAA228V

# ╔═╡ 457a76fb-0d7a-48ee-9405-3685c8281381
using LinearAlgebra

# ╔═╡ b1c2f69a-3e12-40f5-a568-c84537809d64
using Statistics

# ╔═╡ 7fb98a12-b67b-4aad-8489-d062be92c946
using Distributions

# ╔═╡ 3adc20d5-684a-42c4-a4f4-51a641b41cf2
using ProgressLogging

# ╔═╡ f834844a-33b7-4483-a73f-78ab43956d25
using Parameters

# ╔═╡ 0cc2cd07-6de7-4c51-bbf3-a2a84911d0cc
using Random

# ╔═╡ 8c18963c-e916-4dd0-8c1a-9ded8434d1a2
md"""
# Julia and Pluto
AA228V/CS238V: *Validation of Safety-Critical Systems*
"""

# ╔═╡ b3cd799e-17fe-4c80-80e1-107c6e64cc31
PlutoUI.TableOfContents(depth=4)

# ╔═╡ f1a3a270-073b-11eb-1741-37897aa84974
md"""
## Readings/Videos/References

Readings/Videos:
- [Julia Documentation](https://docs.julialang.org/en/v1/)
- [A Brief Introduction to Julia (video)](https://www.youtube.com/watch?v=X4Alzh3QyWU)

Pluto Wiki:
- [Choosing Pluto.jl](https://github.com/fonsp/Pluto.jl/wiki/%F0%9F%92%A1-Choosing-Pluto.jl)
- [Pluto UI](https://github.com/fonsp/Pluto.jl/wiki/%F0%9F%92%BB--UI)
- [Writing and running code](https://github.com/fonsp/Pluto.jl/wiki/%E2%9A%A1-Writing-and-running-code)
- [Basic commands in Pluto](https://github.com/fonsp/Pluto.jl/wiki/%F0%9F%94%8E-Basic-Commands-in-Pluto)
"""

# ╔═╡ 6efb1630-074c-11eb-3186-b3ea3cc6d33b
md"""
# Pluto Notebooks
Pluto, like Jupyter (which stands for "**Ju**lia, **Pyt**hon, **R**"), is a notebook-style environment. We recommend this engaging presentation for a nice introduction: [https://www.youtube.com/watch?v=IAF8DjrQSSk](https://www.youtube.com/watch?v=IAF8DjrQSSk). **We will be using Pluto for the programming projects.**
"""

# ╔═╡ 44fb994a-100c-4009-a064-99cf5b667e64
html"""<div style='display:flex; justify-content:center;'><img alt="Pluto.jl" src="https://raw.githubusercontent.com/fonsp/Pluto.jl/dd0ead4caa2d29a3a2cfa1196d31e3114782d363/frontend/img/logo_white_contour.svg" width=300 height=74 ></div>"""

# ╔═╡ 776449d6-2c4c-450f-a163-21b2290e4a12
md"""
1. Output of a cell is above the code (unlike Jupyter notebooks)
1. Changing a variable in one cell affects other dependent cells (interactive)
1. You cannot redefine a variable or function in a separate cell (due to \#2)
1. Cells contain a single piece of code, or can be wrapped in `begin` `end` blocks
```julia
	begin
		# code goes here
	end
```
5. You can bind interactive objects like sliders to variables using `PlutoUI`
"""

# ╔═╡ 433a69f0-074d-11eb-0698-f9113d9939c3
@bind y Slider(1:10, default=5, show_value=true)

# ╔═╡ 4a4f2872-074d-11eb-0ec8-13783b11ffd7
10y

# ╔═╡ 5541c6ea-20cb-421f-9be9-009aebd80e6c
y^y

# ╔═╡ baf23507-63a7-472e-933c-e100b8cfae51
md"""
# Quick Introduction to Julia
"""

# ╔═╡ 3bf21370-bfda-412b-adbf-dafaf42a471d
html"""<div style='display:flex; justify-content:center;'><img alt="Julia" src="https://julialang.org/assets/infra/logo.svg"></div>"""

# ╔═╡ 991297c0-0776-11eb-082c-e57372352faa
md"""
Julia is a high-level dynamic programming language$^{[\href{https://en.wikipedia.org/wiki/Julia_(programming_language)}{2}]}$ that was designed to solve the two-language problem.

**Two language problem.**$\;\;$One typically uses a high-level language like MATLAB to do scientific computing and create prototypes, but a low-level language like C to implement resulting solutions.

Julia is both fast and easy to prototype in, and supports a wide range of features such as a built-in package manager (so reproducing someone's exact development environment can be done for verification purposes), distributed computing, C and Python interfaces, a powerful REPL (command line, stands for Read, Eval, Print, Loop), and an elegant type system.

The following examples are based on [Learn X in Y Minutes](http://learnxinyminutes.com/docs/julia/). Assumes Julia `v1.11`.

> Note, the output of code is *above* each cell in Pluto notebooks.
"""

# ╔═╡ 0f1aac87-102d-4fec-b012-e364a7a23b0f
y

# ╔═╡ 50caeb00-073c-11eb-36ac-5b8839bb70ab
md"""
## Types

These are different types of numbers.
"""

# ╔═╡ 5996f670-073c-11eb-3a63-67246e676f4e
typeof("hello world")

# ╔═╡ 5fe5f3f0-073c-11eb-33ae-2343a63d952d
typeof(1)

# ╔═╡ 62961fce-073c-11eb-315e-03a405d69157
typeof(1.0)

# ╔═╡ 8ca1d3a0-073c-11eb-2d51-7766203bdf92
supertype(Float64)

# ╔═╡ 90e6d04e-073c-11eb-0a64-a5596e6b6079
supertype(AbstractFloat)

# ╔═╡ 9516be60-073c-11eb-277b-b59f20b2feba
supertype(Real)

# ╔═╡ 9895d300-073c-11eb-1fe4-d3337747efcd
supertype(Number)

# ╔═╡ 9be413a0-073c-11eb-3c73-df78ea75bcd1
supertype(Int64)

# ╔═╡ a10c4462-073c-11eb-31f8-6f675614356d
supertype(Signed)

# ╔═╡ a4de820e-073c-11eb-374e-efd08bbc884c
supertype(Integer)

# ╔═╡ ae65ee42-073c-11eb-1dbd-eb918f086ab7
typeof(true)

# ╔═╡ b427431e-e220-416f-974f-791c43bf0c18
md"""
### Type Hierarchy
"""

# ╔═╡ 92fb38fc-c10d-4a66-9bfe-dfa73f3a7996
AbstractTrees.children(x::Type) = subtypes(x)

# ╔═╡ 9141fbe7-5250-4aff-8754-49a5c9e1884e
print_tree(Environment) # from StanfordAA228V

# ╔═╡ b4ad4aa0-073c-11eb-238e-a913116c0944
md"""
## Boolean Operators

Negation is done with `!`
"""

# ╔═╡ bf0155a2-073c-11eb-1ff2-e9d78baf273a
!true

# ╔═╡ c22997b0-073c-11eb-31c8-952711ee4422
1 == 1

# ╔═╡ c393d612-073c-11eb-071d-93ca1d801f4d
1 != 1

# ╔═╡ c59412e0-073c-11eb-0b89-7d14cda40917
1 > 2 || 4 < 5 # or

# ╔═╡ cd146470-073c-11eb-131c-250b433dbe74
1 > 2 && 4 < 5 # and

# ╔═╡ d5dc2a20-073c-11eb-1667-038dd7a06c63
md"""
Comparisons can be chains. This is equivalent to `1 < 2 && 2 < 3`
"""

# ╔═╡ e05151b0-073c-11eb-2cd2-75f9f350b266
1 < 2 < 3

# ╔═╡ 69601831-6265-4341-beed-4c01a54ed705
1 ≤ 3 ≤ 3 # \le

# ╔═╡ e90af052-709d-4587-9982-f224afd49af2
md"""
`Bool`s can be treated as numbers where `false` maps to `0` and `true` maps to `1`.
"""

# ╔═╡ ddaf4dc2-d5a6-458e-a7f4-280bf64663f3
sum([false, false, true, true, true, true])

# ╔═╡ 4d0b3fd3-a15d-48fc-9219-cd6e02bb5b03
false*100 + true*200

# ╔═╡ e3846930-073c-11eb-384a-b1b2e27d16cc
md"""
## Strings
Use double quotes for strings.
"""

# ╔═╡ ef0693f0-073c-11eb-14da-0bec0f5bfe2e
"This is a string"

# ╔═╡ f131d862-073c-11eb-0584-1947b568926c
typeof("This is a string")

# ╔═╡ f5e1a4d0-073c-11eb-0002-d94d6a932b0b
md"""
**Characters.**$\;$ Use single quotes for characters.
"""

# ╔═╡ 004e40e0-073d-11eb-1475-5dea9854afd4
'a'

# ╔═╡ 781dcaf0-073d-11eb-2498-0f3bdb572f88
typeof('a')

# ╔═╡ 80309880-073d-11eb-10d0-53a70045661f
md"""
Note the 1-based indexing---similar to MATLAB but unlike C/C++/Java/Python.
"""

# ╔═╡ 7a9cd4ae-073d-11eb-1c20-f1a0bc572b33
"This is a string"[1]

# ╔═╡ db37f460-0744-11eb-24b0-cb3767f0cf44
"This is a string"[end]

# ╔═╡ 8ea1a5d0-073d-11eb-029b-01b0401da847
md"""
`$` can be using for "string interpolation".
"""

# ╔═╡ 993a33e0-073d-11eb-2ded-4fc896fd19d7
"2 + 2 = $(2+2)"

# ╔═╡ e95927ed-cfc5-428d-87e1-addc0983d47b
day_of_week = "Wednesday"

# ╔═╡ 4982024e-9e0e-4968-a521-18256e5ead5a
"The current day of the week is $day_of_week"

# ╔═╡ f99743b0-0745-11eb-049e-71c7b72884d1
md"""
## Printing
In Pluto, the last expression is output by default. The use of `print` and `println` will output the result to stdout and it will appear *below* the cell.
"""

# ╔═╡ fbaad713-216b-4557-8132-3a1e95ed0a27
md"""
In the following block, notice only the last expression is output (i.e. sin(π))

*Note: To use multiple lines of code in a single cell, we need to use `begin` ... `end` blocks.*
"""

# ╔═╡ 4a2f40a4-029f-43bb-9f55-3bf001861d0c
begin
	cos(π)
	sin(π)
end

# ╔═╡ d3d5cde4-8d81-45b1-8554-4ee2e009074c
md"""
If we want more outputs, we can add `println` statements or use logging (e.g. `@info`).
"""

# ╔═╡ efa419b0-40ce-41ac-a660-5bd1743b0e6c
begin
	println("cos(π) = ", cos(π))
	println("sin(π) = ", sin(π))
end

# ╔═╡ bc755cd7-40ca-4ad6-9f88-8ff657e4e397
begin
	@info "cos(π)" cos(π)
	@info "sin(π)" sin(π)
end

# ╔═╡ a0ffcbd0-073d-11eb-3c0a-bfc967428073
print(4.5, " is less than ", 5.3)

# ╔═╡ 4968e820-0742-11eb-1b1b-034751f95fb9
println("Welcome to Julia!")

# ╔═╡ c5c2e4a7-6a4f-4d06-9cc5-0011dafbffe3
md"""
You can also use `printf` statements with the `Printf` package
"""

# ╔═╡ b1ed3fa0-b885-48be-9332-653623d4b606
@printf("%.2f is less than %.2f", 4.5, 5.3)

# ╔═╡ 62c9ce6e-0746-11eb-0911-afc23d8b351c
md"""
## Variables
"""

# ╔═╡ 64f36530-0746-11eb-3c66-59091a9c7d7d
v = 5

# ╔═╡ 66335f40-0746-11eb-2f7e-ffe20c76f21f
md"""
Variable names start with a letter, but after that you can use letters, digits, underscores, and exclamation points.
"""

# ╔═╡ 71dc24d0-0746-11eb-1eac-adcbb393c38b
xMarksTheSpot_2Dig! = 1

# ╔═╡ 765e5190-0746-11eb-318e-8954e5e8fa3e
md"""
It is Julian to use lowercase with underscores for variable names.
"""

# ╔═╡ 81956990-0746-11eb-2ca4-63ba1d192b97
x_marks_the_spot_to_dig = 1

# ╔═╡ 85101160-0746-11eb-1501-3101c2006157
md"""
## Arrays
"""

# ╔═╡ 97a54cf0-0746-11eb-391d-d70312796ded
A = Int64[]

# ╔═╡ 9a808070-0746-11eb-05cf-81547eab646d
B = [4, 5, 6]

# ╔═╡ 9d560e9e-0746-11eb-1e55-55e827e7423d
B[1]

# ╔═╡ 9f1264a0-0746-11eb-2554-1f50ba874f57
B[end-1]

# ╔═╡ a2283020-0746-11eb-341a-fb4e280be5d6
matrix = [1 2; 3 4; 5 6]

# ╔═╡ a9ea93c0-0746-11eb-0956-95d12cb066ac
A

# ╔═╡ bc8dd90e-0746-11eb-3c30-1b27e08fd17d
push!(A, 1)

# ╔═╡ ad279650-0746-11eb-2090-81a679e5f3be
push!(A, 2)

# ╔═╡ b1266240-0746-11eb-262c-893974d49c9f
append!(A, B)

# ╔═╡ b5a1b130-0746-11eb-2038-d353aad7e355
A

# ╔═╡ c4f7ee60-0746-11eb-324b-854c3b2e383e
pop!(A)

# ╔═╡ c7484700-0746-11eb-3327-f321a4423d2a
A

# ╔═╡ cb5d3300-0746-11eb-35d3-33280e451394
A[2:4]

# ╔═╡ cd941030-0746-11eb-34d7-216aa0f8f33d
A[2:end]

# ╔═╡ d6663620-0746-11eb-01dc-27b5e3b11ab8
A[2:end-1]

# ╔═╡ da28e370-0746-11eb-1ea1-91664661a74d
push!(A, round(Int64, 1.3))

# ╔═╡ df66e620-0746-11eb-37b9-d3f3ab3dd12f
in(4, A)

# ╔═╡ e3e171c0-0746-11eb-0e1d-c3fc24347d47
4 in A

# ╔═╡ e99fa0f0-0746-11eb-1f5c-7da5d6765131
md"""
You can use $\LaTeX$ keyworks to get unicode characters.
> **Example**: `\in` then hit `<TAB>`.
"""

# ╔═╡ e5afc920-0746-11eb-0558-79599697bec6
4 ∈ A

# ╔═╡ e8dc9f10-0746-11eb-2c56-c9383000043c
!in(4, A)

# ╔═╡ 1b239000-0747-11eb-0a63-d9e58c6dfda3
4 ∉ A # \notin

# ╔═╡ 24b05360-0747-11eb-0783-ab42074819c4
length(A)

# ╔═╡ 410c4e10-0747-11eb-0acf-116ff6073047
md"""
## Tuples
Think of them as immutable arrays.
"""

# ╔═╡ 4842ec70-0747-11eb-37b0-21da3d5049ff
T = (1, 5.4, "hello")

# ╔═╡ 7351b860-0747-11eb-16c5-833309f7fbcb
typeof(T)

# ╔═╡ 750c87c0-0747-11eb-1aeb-d32e03b686f5
T[2]

# ╔═╡ 7f9e37fe-0747-11eb-295e-6d55a31d8395
html"""
This line below gets an <font color='darkred'><b>error</b></font> message. Tuple elements cannot change because they are <i>immutable</i>.
"""

# ╔═╡ 77cd074e-0747-11eb-0306-05ff0e6ada53
T[2] = 3 # can't change elements in a tuple, they are immutable

# ╔═╡ 88232852-0747-11eb-289d-1742e687b041
a, b, c = (1, 2, 3) # you can split out the contents

# ╔═╡ 3589dfc0-0748-11eb-1b7f-672e2f6dcf53
a

# ╔═╡ 48a9a810-0748-11eb-0b15-e9085ebd7b52
b

# ╔═╡ 49c63ba0-0748-11eb-158d-57dcd4ad537d
c

# ╔═╡ 4a742ee0-0748-11eb-364c-f7eb2b89d88a
j, k, l = 1, 2, 3

# ╔═╡ 64a33770-0748-11eb-1221-b994ffb70091
j

# ╔═╡ 668d060e-0748-11eb-0f34-f307c94e755d
k

# ╔═╡ 6722dd70-0748-11eb-31b7-d38b56f4cc0f
l

# ╔═╡ 7226fe92-0748-11eb-215e-49075766b2da
md"""
To create a single-element tuple, you must add the "," at the end
"""

# ╔═╡ 5bfd2a90-0748-11eb-3b5c-8f191bd23f1c
(1,)

# ╔═╡ 7f8e8b20-0748-11eb-39da-435f6c49934a
typeof((1,))

# ╔═╡ 84d7d870-0748-11eb-2a11-5797476719b5
md"""
## Dictionaries
Dictionaries let you map `keys` to `values`.
"""

# ╔═╡ 8dc1a510-0748-11eb-1e2d-ab6fc445d549
dict = Dict()

# ╔═╡ 90afc450-0748-11eb-170e-9fd33246ec06
d = Dict("one"=>1, "two"=>2, "three"=>3)

# ╔═╡ a28f9290-0748-11eb-0e3a-539a124905c0
d["one"]

# ╔═╡ a6f27780-0748-11eb-2ca3-69c3f2923f7e
keys(d)

# ╔═╡ a8c3b510-0748-11eb-39b2-7389d0ee67e4
collect(keys(d))

# ╔═╡ ac35d160-0748-11eb-00c4-e799f1c83746
values(d)

# ╔═╡ b137dc80-0748-11eb-3e3b-d9eb6ade524c
haskey(d, "one")

# ╔═╡ b47d6a92-0748-11eb-1a5c-fb5faaf20c14
haskey(d, 1)

# ╔═╡ bb233aa0-0748-11eb-3488-8d316224bdf8
md"""
## Control Flow
`if` statements, `for` loops, `while` loops, and error catching
"""

# ╔═╡ d2ec8dd0-0748-11eb-298a-5d94d5da2477
some_var = 5

# ╔═╡ db2d2220-0748-11eb-0889-a30e49f2d784
md"""
Here is an `if` statement. Indentation does *not* have a special meaning in Julia..
"""

# ╔═╡ e891a170-0748-11eb-2bbd-e15baa27423c
if some_var > 10
	println("some_var is totally bigger than 10.")
elseif some_var < 10 # This elseif clause is optional.
	println("some_var is smaller than 10.")
else # this else clause is also optional too.
	println("some_var is indeed 10.")
end

# ╔═╡ e129e5c7-74f5-4e5e-8c99-cc99217395c4
md"""
### Ternary if statements
Julia, like many other modern languages, comes with a built-in _ternary if statement_ syntax.

The `?` is the ternary operator with the "else" statement preceeding with ` : ` (spaces are necessary)

```julia
a ? b : c
```

[Link to Julia docs.](https://docs.julialang.org/en/v1/manual/control-flow/#:~:text=The%20so%2Dcalled%20%22ternary%20operator%22)
"""

# ╔═╡ 6f4df9f0-9296-4b8f-a81b-69677cebd6e0
t = 40

# ╔═╡ 1470bb4b-f649-4fc8-8e73-0b04e439f9a2
t > 39 ? "Yes!" : "No."

# ╔═╡ da7ac1f5-ea05-4d88-af31-45f8d11356d2
md"""
Which is equivalent to:
"""

# ╔═╡ f2b5413f-9b8e-4926-a6ee-7b3142f53dcd
if t > 39
	"Yes!"
else
	"No."
end

# ╔═╡ 38ebf397-acfa-4f76-86ea-f6e96cd8fb9a
md"""
In Python, the syntax would be:
```python
"Yes!" if t > 39 else "No."
```
"""

# ╔═╡ 31a01337-d09d-421f-b217-ae85d38562ee
md"""
### For loops
"""

# ╔═╡ 2f09cab2-0749-11eb-3533-79dae3c99545
md"""
`for` loops and `while` loops iterate over an iterable. Iterable types include `Range`, `Array`, `Set`, `Dict`, and `String`.
"""

# ╔═╡ 204a7650-0749-11eb-3bcf-2d2846eb951b
for animal in ["dog", "cat", "mouse"]
	println("$animal is a mammal")
end

# ╔═╡ 50260c90-0749-11eb-0af7-8fa4ca5e3890
for (key, value) in Dict("dog"=>"mammal", "cat"=>"mammal", "sparrow"=>"bird")
	println("$key is a $value")
end

# ╔═╡ 245f5636-a5a7-477b-bf69-b46ad3e07a39
md"""
### While loops
"""

# ╔═╡ 69c88c90-0749-11eb-1c23-a5f0042bf2de
begin
	s = 0
	while s < 4
		println(s)
		s += 1 # shorthand for s = s + 1
	end
end

# ╔═╡ 1c3b59e6-a94b-4d58-8485-5d3cd0285312
md"""
### Try/catch blocks
"""

# ╔═╡ fef5c85a-8536-42aa-b169-4b5ccab6e6bd
md"""
We can also handle errors and execute additional code after catching the error.
"""

# ╔═╡ 870a7a70-0749-11eb-2ab1-e9279dd4642a
try
	error("help!")
	println("We do not make it here after the error.")
catch err
	println("Caught the error! $err")
	println("We do continue in this block after catching an error")
end

# ╔═╡ 9ef92ff2-0749-11eb-119e-35edd2a409c4
md"""
## Functions
"""

# ╔═╡ c3ac5430-0749-11eb-13f0-cde5ec9409cb
md"""
Functions return the value of their last statement (or where you specify `return`)
"""

# ╔═╡ a12781a0-0749-11eb-0019-3154576cfbc5
function add(x, y)
	println("x is $x and y is $y")
	x + y
end

# ╔═╡ 514268b6-e65c-4ba7-b009-c0398e31c890
md"""
In the next block, you'll notice the result of the function's `return` (i.e., `x + y`) displayed above, as per Pluto's usual behavior, and the output of the `println` statement echoed below. 

By default, Julia returns the value of the last expression in a function if the `return` keyword is omitted.
"""

# ╔═╡ ba349f70-0749-11eb-3a4a-294a8c484463
add(5, 6)

# ╔═╡ d6c45450-0749-11eb-3a9c-41cbc41c8d08
md"""
You can define functions with optional positional arguments.
"""

# ╔═╡ dedb4090-0749-11eb-38f4-7dffd22ae8c5
function defaults(a, b, x=5, y=6)
	return "$a $b and $x $y"
end

# ╔═╡ e9fd49f0-0749-11eb-3cf6-b78a95067ee3
defaults('h', 'g')

# ╔═╡ f1339a30-0749-11eb-0fcb-21c6e9917eb9
defaults('h', 'g', 'j')

# ╔═╡ f6631df0-0749-11eb-111d-270176d2bd76
defaults('h', 'g', 'j', 'k')

# ╔═╡ f81b0720-0749-11eb-2217-9dd2714570b3
md"""
You can define functions that take keyword arguments using `;` to separate the variables.
"""

# ╔═╡ 06186b12-074a-11eb-2ff8-d7ecf3b88f3b
function keyword_args(; k1=4, name2="hello")
	return Dict("k1"=>k1, name2=>name2)
end

# ╔═╡ 1ae6cdc0-074a-11eb-2415-710522e2ff61
keyword_args(name2="ness")

# ╔═╡ 1f2031b0-074a-11eb-2f30-43fa1403837e
keyword_args(k1="mine")

# ╔═╡ 2547f820-074a-11eb-2aa5-ffe306f6b1e2
keyword_args()

# ╔═╡ 318a3e2c-d71e-4527-9a97-60bf6fb0a7de
md"""
### Anonymous functions
"""

# ╔═╡ 28c44da0-074a-11eb-07cf-21435e264c3b
md"""
This is "stabby lambda syntax" for creating anonymous functions.
"""

# ╔═╡ 39c7ecb0-074a-11eb-0f6a-59d57a454722
f = x -> x > 2

# ╔═╡ 38f57f03-db11-43b1-becb-d3463143ce8a
md"""
Which is equivalent to:
```julia
function f(x)
	return x > 2
end
```
"""

# ╔═╡ 40e16622-074a-11eb-1f0d-579043dff6df
f(3)

# ╔═╡ 03c4df16-b919-487d-89b2-acff111482e0
md"""
This is similar to `lambda` functions in Python:
```python
f = lambda x: x > 2
```
"""

# ╔═╡ 71088c86-7253-4b0b-b5eb-86533ca98db1
md"""
Another way of defining that a function without using the `function` keyword.
```julia
f(x) = x > 2
```
This creates a _method_ named `f`.
"""

# ╔═╡ de4bc9a3-fcf4-434e-add3-599b4676b267
md"""
### Function returning a function
"""

# ╔═╡ 4602b910-074a-11eb-3b77-dd28974218f6
md"""
This function creates `add` functions. When we call `create_adder` a function is new function is returned.
"""

# ╔═╡ 4e6b95e0-074a-11eb-324c-09c41dd1fb64
function create_adder(x)
	return y -> x + y
end

# ╔═╡ 55249fd0-074a-11eb-1df3-2bfaa49155ae
md"""
You can also name the internal function, if you want.
"""

# ╔═╡ 60e50c10-074a-11eb-0249-339b5ee9bcf2
function create_adder2(x)
	function adder(x)
		x + y
	end
	adder # Julia will automatically return the last statement, even without `return`
end

# ╔═╡ 6cc00530-074a-11eb-08da-5f69fb9e6c08
add10 = create_adder(10)

# ╔═╡ 72681450-074a-11eb-2182-9ba3de8ae5c7
add10(3)

# ╔═╡ 89e51c40-074a-11eb-199f-79a721854c1f
[add10(i) for i in [1, 2, 3]]

# ╔═╡ f1055123-309a-4ee1-aa48-204eb0308b8f
md"""
Functions like `map` and `filter` allow us to apply operations to iterables efficiently.
"""

# ╔═╡ fd41492e-9a13-47d1-8f9f-b55edf4a93f3
map(add10, [1, 2, 3])

# ╔═╡ 4e2c0e4a-64be-4bf6-972a-3f4b1d236fb2
filter(x -> x > 5, [3, 4, 5, 6, 7])

# ╔═╡ e62935c6-8d25-4c2f-a04f-5f7160347d99
md"""
### Use as a conditional distribution
"""

# ╔═╡ ca41086a-8f5e-4617-a4cf-5cf5818ba8bf
md"""
This style of "a function returns another function" is sometimes used in the textbook and projects to return a _conditional distribution_ given the condition as the top-most input.
"""

# ╔═╡ 84b18b58-da13-408b-8e00-ead7265a9bd4
condition = 10

# ╔═╡ 6f1aed26-98a4-4488-b25c-8a1aab4c0050
function p(c)
	return s->Normal(s + c, 1)
end

# ╔═╡ 16693bfd-dc5c-4526-9fff-e8d488638420
md"""
Which represents the conditional:

$$p(s \mid c) = \mathcal{N}(s + c, 1)$$
"""

# ╔═╡ d470ec22-bdff-44bc-ba9c-e829e4d57101
p(10)

# ╔═╡ 5304e5f0-d76b-4b99-bf9d-f865bb41b37e
p(10)(4)

# ╔═╡ 5550f8c3-23c1-4822-82cf-1eaae478f9dd
rand(p(10)(4))

# ╔═╡ 90db2f30-074a-11eb-2990-112df2b43ff3
md"""
## Composite Types

Composite types are user-defined data structures that play a crucial role in enabling multiple dispatch in Julia.
"""

# ╔═╡ 28c3d9f6-8666-44fa-b536-7c05149630eb
struct VehicleState
	position::Vector{Float64}
	velocity::Vector{Float64}
	heading::Float64
	timestamp::Float64
end

# ╔═╡ 755f8671-9076-4f4f-bbbf-1bd214a2d0a9
car_one = VehicleState([0.0, 0.0], [1.0, 0.0], 0.0, 0.0)

# ╔═╡ 6ed95d98-7d68-41a3-9b01-1fee030138cc
car_one.heading # access type properties using dot notation

# ╔═╡ f6771f21-2c0a-40f6-b047-5850e3b2d26e
md"""
In Python, you would achieve this using a _class_.

```python
class VehicleState:
	def __init__(self, position, velocity, heading, timestamp):
		self.position = position
		self.velocity = velocity
		self.heading = heading
		self.timestamp = timestamp

car_one = VehicleState([0.0, 0.0], [1.0, 0.0], 0.0, 0.0)
car_one.heading
```
"""

# ╔═╡ bda58dd0-074a-11eb-37e9-a918c670d380
md"""
Abstract types are just a name used as a point in the type hierarchy.
"""

# ╔═╡ b45a403e-074a-11eb-1144-fd4d939b8bc8
abstract type Cat end

# ╔═╡ 1e3f98f4-8333-47c0-9936-76400633affd
md"""
In Python, this can be done using the _Abstract Base Class_ package.

```python
from abc import ABC

class Cat(ABC):
    pass
```

_Yuck._ 🤢
"""

# ╔═╡ ceb62530-074a-11eb-2d0f-7383bc2bb7ea
subtypes(Number)

# ╔═╡ d37ea9c0-074a-11eb-075b-ef2a3aa472d0
subtypes(Cat)

# ╔═╡ f447e9a0-074a-11eb-1a5b-738a852d47a0
md"""
You can define more constructors for your type. Just define a function of the same name as the type and call an existing constructor to get a value of the correct type.

> `<:` is the subtyping operator.
"""

# ╔═╡ 2754b1f0-0914-11eb-1052-1159804bcc1c
struct Lion <: Cat # Lion is a subtype of Cat
	mane_color
	roar::AbstractString
end

# ╔═╡ 0bb9a6f0-074b-11eb-0cd5-55f6817db5fd
struct Panther <: Cat # Panther is also a subtype of Cat
	eye_color

	# Panters will only have this constructor, and no default constructor
	Panther() = new("green")
end

# ╔═╡ 34216ec0-074b-11eb-0ec5-b933ea8ecf34
subtypes(Cat)

# ╔═╡ 9dde0a7b-bbca-4970-a1ea-db47bbdfa102
md"""
In Python, you can create `Lion` as a class that is a subclass of `Cat`:

```python
class Lion(Cat):
	def __init__(self, mane_color, roar):
		self.mane_color = mane_color
		self.roar = roar
```
"""

# ╔═╡ 573a1ce0-074b-11eb-2c5d-8ddb9d0c07ed
md"""
## Multiple Dispatch
"""

# ╔═╡ 3827f232-0914-11eb-3365-35e127a537ce
function meow(animal::Lion)
	return animal.roar 
end

# ╔═╡ c26ed0e7-1883-4f82-a13c-005def6e78cd
lion = Lion("brown", "ROAAR")

# ╔═╡ 3a2f2748-744b-4a96-beaa-3036a7df7765
md"""
In Python, instead of defining a global method `meow` that takes in a specific type (i.e., Julia's multiple dispatch), you must create the `meow` method as part of the `Lion` class.
```python
class Lion(Cat):
	def __init__(self, mane_color, roar):
		self.mane_color = mane_color
		self.roar = roar

	def meow(self):
		return self.roar

lion = Lion("brown", "ROAAR")
lion.meow()
```

But what if you wanted to add a new method that can operate on `Lion` types?! You'd have to modify the `Lion` class directly...😱
"""

# ╔═╡ 5d31d2a2-074b-11eb-169a-a7423f75a9e6
function meow(animal::Panther)
	return "grrr"
end

# ╔═╡ e534d70a-af4d-4d4b-b844-a5e055af93f2
md"""
We can define functions using our `AbstractType` `Cat` to handle cases where specific implementations for subtypes are missing. This helps catch errors or provide warnings when dealing with multiple subtypes. Julia's multiple dispatch ensures that the most specific method (e.g., `meow(::Panter)`) is invoked first, falling back to the generic method when no specific implementation is found.
"""

# ╔═╡ 9ae8c099-0131-4022-bbc5-c10e78ea3e8d
function meow(cat::Cat)
	@warn "`meow` not defined for the type $(typeof(cat))"
end

# ╔═╡ b2b1a3e2-074b-11eb-1a3d-3fb4f9c09ba9
meow(lion)

# ╔═╡ b7cc6720-074b-11eb-31e0-13dea28d37ec
meow(Panther())

# ╔═╡ da8fe338-68db-4981-af99-07baa5e919bc
struct Ragdoll <: Cat
	cuteness_level::Float64
	name::String
end

# ╔═╡ f8f79331-36ef-4705-93a2-1a5d0da3413e
meow(Ragdoll(Inf, "Sarsa"))

# ╔═╡ 43902359-b6eb-4eb0-aa20-dc3cf913d80e
meow(Panther())

# ╔═╡ 3ab40be6-9b96-42f8-9a06-0fabd73c8a07
md"""
# Packages with Julia

Julia simplifies the process of installing packages, including specific versions, making package management straightforward.

It also allows easy access to packages hosted on GitHub. Within a Pluto notebook, you can type the following to automatically download packages listed in the [Julia Registry](https://juliapackages.com).
"""

# ╔═╡ 203a9c7c-ab82-4ae0-8677-c6ac7b03f73b
md"""
## `StanfordAA228V`
The core library for the programming projects (adapted from the textbook)
"""

# ╔═╡ 9bfecb56-d6d4-4f70-9fb4-a8c0d116136d
md"""
You can [explore the code on GitHub](https://github.com/sisl/StanfordAA228V.jl/tree/main/src).
"""

# ╔═╡ 7cb2d32a-12ed-45ae-aeef-da12ed65eaa6
md"""
## `LinearAlgebra`
Built-in powerful linear algrebra library.
"""

# ╔═╡ f6054e75-fcfe-4e40-a33c-2c3feefbd2ff
norm([1,2,3])

# ╔═╡ 5512d8cd-d9e8-48eb-b720-f6399b226dd6
diagm([1,1,1])

# ╔═╡ 8fa3ecd9-a81a-4789-9ce7-8f2602807643
let
	w = [0.4, 0.4, 0.2]
	x = [1, 2, 3]
	w'x
end

# ╔═╡ cd91a50f-bca9-40ca-a2f4-2e3186d38aee
md"""
Shorthand for `w' * x` or `transpose(w) * x`.
"""

# ╔═╡ 49dcf1dd-536b-446b-9673-5f92dcd7a96c
md"""
## `Statistics`
Built-in library to compute statistics like `mean`, `std`, and `var`.
"""

# ╔═╡ f776a943-79e1-4b69-9501-c188d2435520
md"""
## `Distributions`
Probability distributions like `Uniform`, `Normal`, and `MvNormal`.
"""

# ╔═╡ 44344661-eead-430a-8cd0-99001412fbdc
U = Uniform(90, 95)

# ╔═╡ 659e37a5-8cdc-4a75-b947-7e55f7b2252b
rand(U)

# ╔═╡ d804e977-b6b3-4d82-b840-bbda518b5183
N = Normal(1, 0.1)

# ╔═╡ 42361e99-0cf5-4c2e-b4b9-313556b29941
rand(N)

# ╔═╡ 4b053619-2b4c-421d-bc9b-cd16e6203acc
Mv = MvNormal([0, 0], [1 0; 0 1])

# ╔═╡ 55d0aa5c-b864-4360-acea-5995647b8965
rand(Mv)

# ╔═╡ 29415c3e-1353-4498-b940-07175b3e15d9
md"""
## `ReverseDiff`
Reverse mode automatic differentiation.
"""

# ╔═╡ 58dece25-d00a-4bf5-bbe1-2eb874e842d2
import ReverseDiff: gradient

# ╔═╡ 5d1786e1-94b9-46d8-8f9c-064201eb88a7
let
	f(x) = x[1]^2 + x[2]^3
	gradient(f, [1,1])
end

# ╔═╡ cfb8f437-e897-4688-90b7-c89aa3cf76d9
md"""
## `ProgressLogging`
A nice progress bar for `for` loops.
"""

# ╔═╡ 16735a49-1784-41ce-9653-856f0954ae9a
@progress for i in 1:10
	sleep(0.2)
end

# ╔═╡ c864791d-81fc-4666-87cd-542bc6beb06f
md"""
## `Parameters`
Add default values to structs with `@with_kw`.
"""

# ╔═╡ 1756e1fc-df99-4696-9007-485c807a4b91
@with_kw struct MyParams
	α = 1e-3
	β = 5e-10
	δ = 0.1
end

# ╔═╡ 7c1cc3c2-1447-4d5d-a100-4b232b00ad32
MyParams()

# ╔═╡ 61c633dd-7572-4fea-918d-0f3ec5674822
MyParams(δ=100)

# ╔═╡ c0ecc29f-7760-457b-b290-e8fcb67c7875
md"""
And _many_ more! In your projects, feel free to use as many external Julia packages as you like. 

Running `using PackageName` will automatically install it in the notebook.
"""

# ╔═╡ a1c0fbdd-3ced-4644-8f50-f3dba1e8ec10
md"""
# Working in Pluto
There are some Pluto-specific things you should know about.
"""

# ╔═╡ d1d028eb-9872-4f04-8416-9fa0da0d01ff
md"""
## Errors
You may encounter many errors when using Julia + Pluto (as is the case with any programming language).

### Syntax errors
Syntax errors will always show up in the cell output of the cell where they exists.
"""

# ╔═╡ 4dc6fdc5-3cf1-475d-8f1d-41640dd5dcb4
function this_causes_a_syntax_error(input:Float64) # <- Typing uses :: not :
	return input^2
end

# ╔═╡ 988b8e06-f5c0-4882-a50f-d96e5d218310
md"""
### Runtime errors
While runtime errors will show up in the cell _where the function is run_.
"""

# ╔═╡ 0e51c879-fc04-4af0-8325-01a55c4e9082
function no_syntax_errors_here(input::Float64)
	return Int(input)
end

# ╔═╡ c68b68e9-078d-4300-beba-49d37206902b
md"""
_**Here's some long text to separate the cells a bit...**_

Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed vitae ipsum sed nulla ultricies dictum sed vel urna. Vivamus et justo et dui euismod consectetur. Curabitur lobortis viverra mauris. Nullam malesuada lorem nulla, et eleifend orci venenatis ac. Maecenas varius rutrum facilisis. Vestibulum sed efficitur nunc. Fusce non elementum lacus. Pellentesque facilisis turpis ut mauris sodales, sit amet sollicitudin dolor tincidunt. Aliquam eget tempor leo. Fusce sollicitudin purus est, eget sodales sem tincidunt ornare. Maecenas eget mauris ac justo tincidunt tincidunt. Maecenas luctus non odio quis consequat.

Maecenas ac quam ac quam mollis consectetur. Mauris vitae semper nisi, nec imperdiet mauris. Nunc ex sapien, scelerisque vel augue in, feugiat tempus est. Aliquam maximus arcu sed velit tempus, sed suscipit diam ullamcorper. Morbi id dapibus ipsum. Cras sed ligula nibh. Nam imperdiet justo eu eros ornare dictum. Aliquam sapien sem, pharetra pretium tortor ut, vehicula pulvinar magna. In luctus pharetra leo in gravida. Ut dignissim nibh non lacus interdum pellentesque. Aenean eu dolor non purus sodales faucibus ac id est. Sed id luctus lacus, non auctor nunc. Integer nec massa ornare, tincidunt neque sit amet, auctor mauris.

Donec eu dui vel leo tempor scelerisque non et lectus. Donec congue tellus sapien. Praesent nec dignissim ligula. Fusce eget quam eget dui iaculis congue. Fusce fermentum mattis pretium. Nunc hendrerit massa magna, id condimentum nibh faucibus eget. Integer ut orci sit amet orci rhoncus maximus eget quis magna. Sed egestas diam nec neque mollis egestas. Nullam magna risus, semper vel nulla id, sagittis fringilla quam. Aliquam lobortis tristique nibh, ac congue felis porttitor lobortis.
"""

# ╔═╡ 65cc6f91-d273-4134-9bcb-2485b95dd1d9
no_syntax_errors_here(1.5)

# ╔═╡ e026d4f7-6ed8-43fa-b146-7cc3658ae372
md"""
So you may not see errors in the cell where the error exists, but Pluto will point you to the right place.

See here that it points to our function `no_syntax_errors_here` as `line 2`, you can click on that and it will jump to the line where the error occurred.
"""

# ╔═╡ 75bd1e1a-30f4-479b-829d-3a802a12a0f5
md"""
### Dependent cell errors
Say one cell defines a variable and another cell uses it. If the cell that defines it has errors, then the other cell will be temporarily disabled and will tell you why.
"""

# ╔═╡ 80d610cf-f2d8-4c6f-993c-a83c180a7d43
issue_variable = Int("string")

# ╔═╡ c3a7e709-4fdf-429d-8217-ec65abf9b350
md"""
_**Here's some long text to separate the cells a bit...**_

Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed vitae ipsum sed nulla ultricies dictum sed vel urna. Vivamus et justo et dui euismod consectetur. Curabitur lobortis viverra mauris. Nullam malesuada lorem nulla, et eleifend orci venenatis ac. Maecenas varius rutrum facilisis. Vestibulum sed efficitur nunc. Fusce non elementum lacus. Pellentesque facilisis turpis ut mauris sodales, sit amet sollicitudin dolor tincidunt. Aliquam eget tempor leo. Fusce sollicitudin purus est, eget sodales sem tincidunt ornare. Maecenas eget mauris ac justo tincidunt tincidunt. Maecenas luctus non odio quis consequat.
"""

# ╔═╡ 5cd78123-0312-472e-bfa3-9e166e54b89f
issue_variable^2

# ╔═╡ c9a1900c-3a99-47e7-ad2e-9eb98504a3c9
md"""
## Running cells (keyboard shortcuts)
After editing a cell, you can run it several ways:
1. To run the current cell: `⇧+Enter`
    - This will also rerun a cell if your cursor is focused on it.
1. To run the current cell and create a new one below: `⌃+Enter` or `⌘+Enter`
    - This will _not_ rerun the current cell if has already been run (indicated by the cell color)
1. To run all cells with unsaved changes: `⌃+S` or `⌘+S`
    - See in the top right of the notebook when this option is available.
"""

# ╔═╡ 7d42e44e-7a4f-4325-aec8-ecda3bd2a751


# ╔═╡ bee6191c-d553-44e6-952b-ed1744d735a0
begin
	#=
	See what you're missing?! There could be so much in here.
	=#
	
	😅 = "yes you can use emojis as variable names" # \:sweat_smile:
	🍕 = "is arguable the best" # \:pizza:
	🔥 = "this class" # \:fire:
	
	md"""
	## Hiding/showing cells
	Pluto allows you to easily hide and show cells to clean up the notebook.

	Click the "eye" icon in the top-left of the cell to toggle it's code visibility.
	"""
end

# ╔═╡ 89d08f17-9fc0-4665-a500-ac5967deedc5
md"""
## Disabling cells
You can also disable a cell by clicking the "cirlce with ⋯" icon in the top-right of the cell.
"""

# ╔═╡ 1a4a6777-c031-40bb-a01a-0e59226f22b3
# ╠═╡ disabled = true
#=╠═╡
lets_disable_this = 12345
  ╠═╡ =#

# ╔═╡ 2d1b88f7-91dd-490f-99c2-22731313fab7
#=╠═╡
relies_on_the_above = lets_disable_this * 6789
  ╠═╡ =#

# ╔═╡ 8d6ddcfa-6e72-4a5f-a8d7-bbeacd6b470b
md"""
## Suppressing cell outputs
To suppress the output of a cell from showing up, you can add a semicolon `;` to the end of the expression.
"""

# ╔═╡ c8748922-8ca5-486e-93ae-8d730e4a7c69
sqrt(12345);

# ╔═╡ 400a841e-e742-4b6f-9277-1648b58079e9
sqrt(12345)

# ╔═╡ 0cebdf5d-e2e3-4131-a623-79581b22898f
md"""
If you're using a `begin`/`end` block, you will need to put it at after the `end`.
"""

# ╔═╡ add31696-8a6d-4808-8ac8-91429768cab0
begin
	sqrt(12345)
end;

# ╔═╡ 34629b2d-defe-4bd1-8d29-ab59c4f7299b
begin
	sqrt(12345); # This won't suppress it
end

# ╔═╡ a7aa0880-d88a-42d8-9cf9-892210486aef
md"""
# More Advanced Julia
We'll show off some more advanced Julia syntax that we use in the textbook and projects.
"""

# ╔═╡ 89b956a9-05ec-4120-9469-468270315881
md"""
## Method overloading with dot notation
Say a package, `StanfordAA228V` in this example, defines a function called `depth` that takes in some type, a `NominalTrajectoryDistribution` in this case. If we want to add another method for `depth` to allow multiple dispatch to work on our added type, we have to use the dot notation to **add that method to the package in question.
"""

# ╔═╡ 29a7d282-7235-447f-bd6d-3f5b6f4df57c
struct MyType
	d
end

# ╔═╡ edeaff05-3f37-41f2-b10e-7530c0bbb301
function StanfordAA228V.depth(mt::MyType)
	return mt.d
end

# ╔═╡ 0f13d11c-b580-47cd-8356-93fead9e9f2f
mt = MyType(41)

# ╔═╡ d84184b9-b9bb-4552-9471-f22931242aa5
depth(mt)

# ╔═╡ d4243a61-d0a0-4f07-8a02-abcbda51a5d8
md"""
If you don't do this, Julia will completely override the `depth` methods with your new method and this will cause some confusing errors (note, we catch this for you in the project notebooks and give you a helpful error in that case).
"""

# ╔═╡ 448a2014-33ab-4043-aa5c-aa1a6a763299
P = NominalTrajectoryDistribution(Normal(), missing, 10)

# ╔═╡ e56132c3-cd06-4f3c-9e18-31b30b39163c
depth(P)

# ╔═╡ f6c78515-7efe-429f-94df-9413f6974410
md"""
In the projects, you will likely run into this for the following methods:
- `initial_state_distribution`
- `disturbance_distribution`
- `depth`

This is common in Julia where you need to use the funciton name qualified with the module/package name. Read more in the ["Namespace Management" section of the Julia docs.](https://docs.julialang.org/en/v1/manual/modules/#namespace-management)
"""

# ╔═╡ 49e1b7c3-7ae0-480f-acc6-0ec08d91564e
md"""
## Function-like objects
You can make an arbitrary composite type "callable" using the following syntax.

Reference: [Julia docs: Function-like objects](https://docs.julialang.org/en/v1/manual/methods/#Function-like-objects)
"""

# ╔═╡ fc607c2c-e153-461b-9e00-e7e478e3266f
begin
	struct GaussianSensor
		Do::Function # distribution = Do(s)
	end

	# Function-like object
	(gs::GaussianSensor)(s) = rand(gs.Do(s))
end

# ╔═╡ 92f6133c-1e19-4b96-b43b-58bc67fbcfe8
Do = s->Normal(s, 1)

# ╔═╡ 822610e1-d90b-46d0-a78b-7661cb492299
sensor = GaussianSensor(Do)

# ╔═╡ d5318314-8569-4652-a2b7-20eca1c7c6cd
sensor(10)

# ╔═╡ 5808f5d3-8bbd-4df6-8502-0e40335922e4
sensor(50)

# ╔═╡ edc61fc4-5123-439e-ab0d-29fc8cee5cf2
md"""
This is similar to Python's `__call__` method on a class if you're used to that.

```python
from scipy.stats import norm

class GaussianSensor:
    def __init__(self, Do):
        self.Do = Do

    def __call__(self, s):
        return self.Do(s).rvs()

Do = lambda s: norm(loc=s, scale=1)

sensor = GaussianSensor(Do)
sensor(10)
```
"""

# ╔═╡ cd775016-1af0-494b-b8d6-abc98afd8cae
md"""
## Random seeding
You can control the _random number generator_ (RNG) seed using the `Random` package, which comes pre-installed with the Julia base library.
"""

# ╔═╡ 2e5fc949-bab1-42ab-9543-0fb7342ec6c6
begin
	Random.seed!(0)
	rand() # random number uniform between [0, 1]
end

# ╔═╡ cb05d040-c5a8-44e9-8e98-a462452cdd09
html"""
<h2>Global variables</h2>
<p>In Pluto, global variables are <b><u>underlined</u></b>.</p>

<p>You can <b><code>⌘-Click</code></b> (on macOS) or <b><code>⌃-click</code></b> (on Linux/Windows) to jump to the definition of the variable (hover over the variable to get a tooltip if you forget this keyboard shortcut).</p>
"""

# ╔═╡ 47e9e011-1927-470b-a1e3-6ebd866f4b3b
sensor

# ╔═╡ 73b267d5-1eb6-48da-ae75-65a21ae64e40
md"""
## Locally scoped cells
Use Julia's `let` block, we can create locally scoped variables (e.g., `a`, `b`, and `c`).
"""

# ╔═╡ 1e7f5155-9bf4-41ca-8d0d-337828a7466c
value = let
	a = 100
	b = 10
	c = 1
	a + b + c
end

# ╔═╡ 150db79c-d2ac-4227-949b-8f448aad7703
md"""
The last line of code in the `let` block will be returned and assigned to the globally scoped `value` variable in this case (notice no `return` keyword on purpose).

This way, you can reuse variable names such as `a`, `b`, and `c` without affecting other cells that may also use those name in global scope.

Two other ways to do this.
1. Just define a new function:
```julia
function my_value()
	a = 100
	b = 10
	c = 1
	return a + b + c
end

value = my_value()
```
2. Use the `local` keyword within a `begin`/`end` block:
```julia
begin
	local a = 100
	local b = 10
	local c = 1
	value = a + b + c
end
```
"""

# ╔═╡ 6ffac9f1-7053-4b9a-812c-1f483685e7f0
md"""
## List comprehension
A syntactic convenience in most modern languages comes with _for loop list comprehension_. 
"""

# ╔═╡ dab00858-fa79-46fb-8472-262b18f9c2da
F = [x^2 for x in 1:10]

# ╔═╡ b6188b39-b3a4-4a96-8f82-b624abe99029
md"""
This is equivalant to:
```julia
F = []
for x in 1:10
	push!(F, x^2)
end
```
"""

# ╔═╡ f8a0bc8f-5f31-40da-8964-29113cae9bae
md"""
This can also be done over multiple for loops.
"""

# ╔═╡ 30a336af-8d9f-448e-b763-d4dbec4f68a5
D = [x + y for x in 1:4 for y in 5:8]

# ╔═╡ ba74f9d2-cf25-4f8a-99cb-7ba3ad7fa594
md"""
This is equivalant to:
```julia
D = []
for x in 1:4
	for y in 5:8
		push!(D, x + y)
	end
end 
```
"""

# ╔═╡ 2ee67812-ecb2-4d53-a890-ea398e88e36c
md"""
You can also create matrices using commas in list comprehensions like so:
"""

# ╔═╡ d8a738c6-6264-48d1-8aa4-6147ee2b7898
M = [x + y for x in 1:4, y in 5:8]

# ╔═╡ da4027dc-67e7-4297-9804-723c95296704
md"""
This is equivalent to:
```julia
M = []
for x in 1:4
    row = []
    for y in 5:8
        push!(row, x + y)
    end
    push!(M, row)
end
M = hcat(M...)  # Convert the nested array to a 2D matrix
```
"""

# ╔═╡ 2f2a0f8b-53b5-4283-b2b6-8735e4071b68
md"""
To see it more clearly, here we just place the `(x,y)` values as the elements of the matrix.
"""

# ╔═╡ 959910e8-02ce-4b31-bd0d-5aa5ec3cc03f
MC = [(x,y) for x in 1:4, y in 5:8]

# ╔═╡ d7a032b6-5023-436a-8407-5e6e7684a241
html"""
<p>Notice that <b><u><code>y</b></u></code> is incorrectly underlined. Even though that variable name <i>is</i> defined as a global, we can see that it's actually a local variable within the for loop.</p>

<p>Sometimes Pluto gets confused with globals 👉👈</p>
"""

# ╔═╡ 5dd79519-5522-40b2-965c-603e28d1126d
md"""
Functions that take in an array can handle list comprehensions without the outer square brackets.
"""

# ╔═╡ f78c40e9-8299-4397-9052-9390805b0193
sum([x + y for x in 1:4 for y in 5:8])

# ╔═╡ 42226951-148b-4aa9-979c-8f8bef22ce6f
sum(x + y for x in 1:4 for y in 5:8)

# ╔═╡ cfb583d9-af6e-4129-bae7-0aaa513ded3d
md"""
## Mapping and broadcasting
You can map a function over each element in a vector using `map`.
"""

# ╔═╡ f9b050ae-2b92-4274-8d35-15bb29ad0162
map(sqrt, [1,2,3,4])

# ╔═╡ 21bc516e-8000-4e5c-93c6-e1b9a3ec782e
md"""
Or you could use _broadcasting_ which uses a dot `.` to apply/map the function over each element, just like `map`. Broascasting in Julia is a powerful convenience that we use a lot in the textbook.
"""

# ╔═╡ b23a65ba-5ebc-48bc-b013-539af1d28b65
sqrt.([1,2,3,4])

# ╔═╡ 7fc71f67-0b41-4dda-abf1-ba6db3d027b4
md"""
## Splatting...
The concept of "splatting" in Julia (using the `...` notation) allows for you to expand, or _splat_, each element of a vector or tuple to be treated as individual arguments to a function.
"""

# ╔═╡ 6d34e3cd-8ca0-44fe-852d-3d09922802dc
function some_function(a, b, c)
	println("My favorite candies are $a, $b, and $c.")
end

# ╔═╡ b1b0037a-f058-4bac-9d8f-038260c0eff1
candies = ["Sour Patch Kids", "Sour Punch Straws", "Sugar Lips"]

# ╔═╡ f07431f4-984f-4882-8962-3cbbeaa1007a
some_function(candies...)

# ╔═╡ 2d5920cb-c879-44e8-855d-8bd574c0f4bb
md"""
Where simply passing the vector as the input without splatting would cause an error.
"""

# ╔═╡ 215be802-1998-4887-86a6-3bb8e6e5128c
some_function(candies)

# ╔═╡ 4bbe0e01-b27d-4247-b294-42892db75702
md"""
## Underscore as digit separator
You can use the underscore `_` as a convenient digit separator: [Link to Julia docs](https://docs.julialang.org/en/v1/manual/integers-and-floating-point-numbers/#:~:text=The%20underscore).
"""

# ╔═╡ b0e214ce-7283-4f97-a956-cd218473762f
1000000 == 1_000_000

# ╔═╡ 15403d6b-6e2b-488e-806a-9e9d7ebbfb94
100000 == 100_000

# ╔═╡ c0ff7327-ea30-4d74-9f17-41309b1abf40
10000 == 10_000

# ╔═╡ 134c7eb8-e52b-4dc3-bdcd-fb8bb0a6ecfd
1000 == 1_000

# ╔═╡ 87682e53-7286-4a17-9ee4-2f725cd508a0
md"""
You can use them arbitrarily, but that's just weird...
"""

# ╔═╡ cb3646a1-4d12-4087-8fad-18e44c357623
1234567 == 12_3_456_7

# ╔═╡ 3637fed7-5acc-49ad-8ec0-b7b48ea91d55
md"""
## Unicode symbols
You can use Unicode—and even emojis 🙃—as variable and function names. Here are some common ones we use throughout this course:

| Unicode | Code |
|:-------:|:----:|
| `τ` | `\tab` |
| `ψ` | `\psi` |
| `ℓ` | `\ell` |
| `π` | `\pi` |
| `σ` | `\sigma` |
| `Σ` | `\Sigma` |
| `θ` | `\theta` |
| `ω` | `\omega` |
| `ℛ` | `\scrR` |
| `𝔼` | `\bbE` |
| `²` | `\^2` |
| `₂` | `\_2` |
| `🍕` | `\:pizza:` |

To enter them into cells, type the above "**Code**" and hit `<TAB><TAB>` (or `<TAB><ENTER>`). Feel free to use any Unicode/emojis to your hearts desire 💙

Most of the unicode characters follow $\LaTeX$ names, such as `\square` for `□`, `\lozenge` for `◊`, and `\sqrt` for `√`.

See the Julia docs for more examples: [https://docs.julialang.org/en/v1/manual/unicode-input/](https://docs.julialang.org/en/v1/manual/unicode-input/)
"""

# ╔═╡ f191673f-236f-44a0-a7d6-e47264c4b6c1
md"""
---
"""

# ╔═╡ ed3b1a88-930f-4b3f-aadf-a29da0c7d953
begin
	import StanfordAA228V: highlight, notebook_style
	notebook_style()
end

# ╔═╡ 0d994f6e-f874-4d7b-8335-a2e09184aadd
highlight(md"""Don't spin your wheels if you get super stuck/frustrated, ask us for help on [Ed](https://edstem.org/us/courses/69226/discussion)!

We're more than happy to help.""")

# ╔═╡ fa03914b-1603-455c-a437-d17eda6fcee2
Markdown.MD(md"""
## New Pluto cells
You can create as many new cells anywhere as you like in your project notebooks. Click the `+` icon on the right or hit `⌘+Enter` (on macOS) or `⌃+Enter` (on Linux/Windows) within an existing cell to create a new one below it.
""",
highlight(md"""**Important**: Please do not modify/delete any existing cells in the project notebooks."""))

# ╔═╡ 2e29d581-7281-41c4-a079-79f951a6f0ec
highlight(md"**Important**: Please do not disable any existing cell in the project notebooks.")

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
AbstractTrees = "1520ce14-60c1-5f80-bbc7-55ef81b5835c"
Distributions = "31c24e10-a181-5473-b8eb-7969acd0382f"
LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
Parameters = "d96e819e-fc66-5662-9728-84c9c7592b0a"
PlutoUI = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
Printf = "de0858da-6303-5e67-8744-51eddeeeb8d7"
ProgressLogging = "33c8b6b6-d38a-422a-b730-caa89a2f386c"
Random = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"
ReverseDiff = "37e2e3b7-166d-5795-8a7a-e32c996b4267"
StanfordAA228V = "6f6e590e-f8c2-4a21-9268-94576b9fb3b1"
Statistics = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"

[compat]
AbstractTrees = "~0.4.5"
Distributions = "~0.25.116"
Parameters = "~0.12.3"
PlutoUI = "~0.7.60"
ProgressLogging = "~0.1.4"
ReverseDiff = "~1.15.3"
StanfordAA228V = "~0.1.22"
Statistics = "~1.11.1"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.11.2"
manifest_format = "2.0"
project_hash = "c72bb9eea46ddf53ee32e4c95acff684a1988159"

[[deps.AbstractFFTs]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "d92ad398961a3ed262d8bf04a1a2b8340f915fef"
uuid = "621f4979-c628-5d54-868e-fcf4e3e8185c"
version = "1.5.0"
weakdeps = ["ChainRulesCore", "Test"]

    [deps.AbstractFFTs.extensions]
    AbstractFFTsChainRulesCoreExt = "ChainRulesCore"
    AbstractFFTsTestExt = "Test"

[[deps.AbstractPlutoDingetjes]]
deps = ["Pkg"]
git-tree-sha1 = "6e1d2a35f2f90a4bc7c2ed98079b2ba09c35b83a"
uuid = "6e696c72-6542-2067-7265-42206c756150"
version = "1.3.2"

[[deps.AbstractTrees]]
git-tree-sha1 = "2d9c9a55f9c93e8887ad391fbae72f8ef55e1177"
uuid = "1520ce14-60c1-5f80-bbc7-55ef81b5835c"
version = "0.4.5"

[[deps.Adapt]]
deps = ["LinearAlgebra", "Requires"]
git-tree-sha1 = "50c3c56a52972d78e8be9fd135bfb91c9574c140"
uuid = "79e6a3ab-5dfb-504d-930d-738a2a938a0e"
version = "4.1.1"
weakdeps = ["StaticArrays"]

    [deps.Adapt.extensions]
    AdaptStaticArraysExt = "StaticArrays"

[[deps.AliasTables]]
deps = ["PtrArrays", "Random"]
git-tree-sha1 = "9876e1e164b144ca45e9e3198d0b689cadfed9ff"
uuid = "66dad0bd-aa9a-41b7-9441-69ab47430ed8"
version = "1.1.3"

[[deps.ArgTools]]
uuid = "0dad84c5-d112-42e6-8d28-ef12dabb789f"
version = "1.1.2"

[[deps.ArrayInterface]]
deps = ["Adapt", "LinearAlgebra"]
git-tree-sha1 = "017fcb757f8e921fb44ee063a7aafe5f89b86dd1"
uuid = "4fba245c-0d91-5ea0-9b3e-6abc04ee57a9"
version = "7.18.0"

    [deps.ArrayInterface.extensions]
    ArrayInterfaceBandedMatricesExt = "BandedMatrices"
    ArrayInterfaceBlockBandedMatricesExt = "BlockBandedMatrices"
    ArrayInterfaceCUDAExt = "CUDA"
    ArrayInterfaceCUDSSExt = "CUDSS"
    ArrayInterfaceChainRulesCoreExt = "ChainRulesCore"
    ArrayInterfaceChainRulesExt = "ChainRules"
    ArrayInterfaceGPUArraysCoreExt = "GPUArraysCore"
    ArrayInterfaceReverseDiffExt = "ReverseDiff"
    ArrayInterfaceSparseArraysExt = "SparseArrays"
    ArrayInterfaceStaticArraysCoreExt = "StaticArraysCore"
    ArrayInterfaceTrackerExt = "Tracker"

    [deps.ArrayInterface.weakdeps]
    BandedMatrices = "aae01518-5342-5314-be14-df237901396f"
    BlockBandedMatrices = "ffab5731-97b5-5995-9138-79e8c1846df0"
    CUDA = "052768ef-5323-5732-b1bb-66c8b64840ba"
    CUDSS = "45b445bb-4962-46a0-9369-b4df9d0f772e"
    ChainRules = "082447d4-558c-5d27-93f4-14fc19e9eca2"
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    GPUArraysCore = "46192b85-c4d5-4398-a991-12ede77f4527"
    ReverseDiff = "37e2e3b7-166d-5795-8a7a-e32c996b4267"
    SparseArrays = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
    StaticArraysCore = "1e83bf80-4336-4d27-bf5d-d5a4f845583c"
    Tracker = "9f7883ad-71c0-57eb-9f7f-b5c9e6d3789c"

[[deps.Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"
version = "1.11.0"

[[deps.Atomix]]
deps = ["UnsafeAtomics"]
git-tree-sha1 = "c3b238aa28c1bebd4b5ea4988bebf27e9a01b72b"
uuid = "a9b6321e-bd34-4604-b9c9-b65b8de01458"
version = "1.0.1"

    [deps.Atomix.extensions]
    AtomixCUDAExt = "CUDA"
    AtomixMetalExt = "Metal"
    AtomixoneAPIExt = "oneAPI"

    [deps.Atomix.weakdeps]
    CUDA = "052768ef-5323-5732-b1bb-66c8b64840ba"
    Metal = "dde4c033-4e86-420c-a63e-0dd931031962"
    oneAPI = "8f75cd03-7ff8-4ecb-9b8f-daf728133b1b"

[[deps.BSON]]
git-tree-sha1 = "4c3e506685c527ac6a54ccc0c8c76fd6f91b42fb"
uuid = "fbb218c0-5317-5bc6-957e-2ee96dd4b1f0"
version = "0.3.9"

[[deps.Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"
version = "1.11.0"

[[deps.BenchmarkTools]]
deps = ["Compat", "JSON", "Logging", "Printf", "Profile", "Statistics", "UUIDs"]
git-tree-sha1 = "e38fbc49a620f5d0b660d7f543db1009fe0f8336"
uuid = "6e4b80f9-dd63-53aa-95a3-0cdb28fa8baf"
version = "1.6.0"

[[deps.BitFlags]]
git-tree-sha1 = "0691e34b3bb8be9307330f88d1a3c3f25466c24d"
uuid = "d1d4a3ce-64b1-5f1a-9ba4-7e7e69966f35"
version = "0.1.9"

[[deps.Bzip2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "8873e196c2eb87962a2048b3b8e08946535864a1"
uuid = "6e34b625-4abd-537c-b88f-471c36dfa7a0"
version = "1.0.8+4"

[[deps.CEnum]]
git-tree-sha1 = "389ad5c84de1ae7cf0e28e381131c98ea87d54fc"
uuid = "fa961155-64e5-5f13-b03f-caf6b980ea82"
version = "0.5.0"

[[deps.CRlibm]]
deps = ["CRlibm_jll"]
git-tree-sha1 = "32abd86e3c2025db5172aa182b982debed519834"
uuid = "96374032-68de-5a5b-8d9e-752f78720389"
version = "1.0.1"

[[deps.CRlibm_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "e329286945d0cfc04456972ea732551869af1cfc"
uuid = "4e9b3aee-d8a1-5a3d-ad8b-7d824db253f0"
version = "1.0.1+0"

[[deps.Cairo_jll]]
deps = ["Artifacts", "Bzip2_jll", "CompilerSupportLibraries_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "JLLWrappers", "LZO_jll", "Libdl", "Pixman_jll", "Xorg_libXext_jll", "Xorg_libXrender_jll", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "009060c9a6168704143100f36ab08f06c2af4642"
uuid = "83423d85-b0ee-5818-9007-b63ccbeb887a"
version = "1.18.2+1"

[[deps.ChainRules]]
deps = ["Adapt", "ChainRulesCore", "Compat", "Distributed", "GPUArraysCore", "IrrationalConstants", "LinearAlgebra", "Random", "RealDot", "SparseArrays", "SparseInverseSubset", "Statistics", "StructArrays", "SuiteSparse"]
git-tree-sha1 = "4312d7869590fab4a4f789e97bd82f0a04eaaa05"
uuid = "082447d4-558c-5d27-93f4-14fc19e9eca2"
version = "1.72.2"

[[deps.ChainRulesCore]]
deps = ["Compat", "LinearAlgebra"]
git-tree-sha1 = "1713c74e00545bfe14605d2a2be1712de8fbcb58"
uuid = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
version = "1.25.1"
weakdeps = ["SparseArrays"]

    [deps.ChainRulesCore.extensions]
    ChainRulesCoreSparseArraysExt = "SparseArrays"

[[deps.CodecBzip2]]
deps = ["Bzip2_jll", "TranscodingStreams"]
git-tree-sha1 = "84990fa864b7f2b4901901ca12736e45ee79068c"
uuid = "523fee87-0ab8-5b00-afb7-3ecf72e48cfd"
version = "0.8.5"

[[deps.CodecZlib]]
deps = ["TranscodingStreams", "Zlib_jll"]
git-tree-sha1 = "bce6804e5e6044c6daab27bb533d1295e4a2e759"
uuid = "944b1d66-785c-5afd-91f1-9de20f533193"
version = "0.7.6"

[[deps.ColorSchemes]]
deps = ["ColorTypes", "ColorVectorSpace", "Colors", "FixedPointNumbers", "PrecompileTools", "Random"]
git-tree-sha1 = "c785dfb1b3bfddd1da557e861b919819b82bbe5b"
uuid = "35d6a980-a343-548e-a6ea-1d62b119f2f4"
version = "3.27.1"

[[deps.ColorTypes]]
deps = ["FixedPointNumbers", "Random"]
git-tree-sha1 = "b10d0b65641d57b8b4d5e234446582de5047050d"
uuid = "3da002f7-5984-5a60-b8a6-cbb66c0b333f"
version = "0.11.5"

[[deps.ColorVectorSpace]]
deps = ["ColorTypes", "FixedPointNumbers", "LinearAlgebra", "Requires", "Statistics", "TensorCore"]
git-tree-sha1 = "a1f44953f2382ebb937d60dafbe2deea4bd23249"
uuid = "c3611d14-8923-5661-9e6a-0046d554d3a4"
version = "0.10.0"
weakdeps = ["SpecialFunctions"]

    [deps.ColorVectorSpace.extensions]
    SpecialFunctionsExt = "SpecialFunctions"

[[deps.Colors]]
deps = ["ColorTypes", "FixedPointNumbers", "Reexport"]
git-tree-sha1 = "64e15186f0aa277e174aa81798f7eb8598e0157e"
uuid = "5ae59095-9a9b-59fe-a467-6f913c188581"
version = "0.13.0"

[[deps.CommonSubexpressions]]
deps = ["MacroTools"]
git-tree-sha1 = "cda2cfaebb4be89c9084adaca7dd7333369715c5"
uuid = "bbf7d656-a473-5ed7-a52c-81e309532950"
version = "0.3.1"

[[deps.Compat]]
deps = ["TOML", "UUIDs"]
git-tree-sha1 = "8ae8d32e09f0dcf42a36b90d4e17f5dd2e4c4215"
uuid = "34da2185-b29b-5c13-b0c7-acf172513d20"
version = "4.16.0"
weakdeps = ["Dates", "LinearAlgebra"]

    [deps.Compat.extensions]
    CompatLinearAlgebraExt = "LinearAlgebra"

[[deps.CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"
version = "1.1.1+0"

[[deps.ConcurrentUtilities]]
deps = ["Serialization", "Sockets"]
git-tree-sha1 = "f36e5e8fdffcb5646ea5da81495a5a7566005127"
uuid = "f0e56b4a-5159-44fe-b623-3e5288b988bb"
version = "2.4.3"

[[deps.Configurations]]
deps = ["ExproniconLite", "OrderedCollections", "TOML"]
git-tree-sha1 = "4358750bb58a3caefd5f37a4a0c5bfdbbf075252"
uuid = "5218b696-f38b-4ac9-8b61-a12ec717816d"
version = "0.17.6"

[[deps.ConstructionBase]]
git-tree-sha1 = "76219f1ed5771adbb096743bff43fb5fdd4c1157"
uuid = "187b0558-2788-49d3-abe0-74a17ed4e7c9"
version = "1.5.8"

    [deps.ConstructionBase.extensions]
    ConstructionBaseIntervalSetsExt = "IntervalSets"
    ConstructionBaseLinearAlgebraExt = "LinearAlgebra"
    ConstructionBaseStaticArraysExt = "StaticArrays"

    [deps.ConstructionBase.weakdeps]
    IntervalSets = "8197267c-284f-5f27-9208-e0e47529a953"
    LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
    StaticArrays = "90137ffa-7385-5640-81b9-e52037218182"

[[deps.Contour]]
git-tree-sha1 = "439e35b0b36e2e5881738abc8857bd92ad6ff9a8"
uuid = "d38c429a-6771-53c6-b99e-75d170b6e991"
version = "0.6.3"

[[deps.DataAPI]]
git-tree-sha1 = "abe83f3a2f1b857aac70ef8b269080af17764bbe"
uuid = "9a962f9c-6df0-11e9-0e5d-c546b8b5ee8a"
version = "1.16.0"

[[deps.DataStructures]]
deps = ["Compat", "InteractiveUtils", "OrderedCollections"]
git-tree-sha1 = "1d0a14036acb104d9e89698bd408f63ab58cdc82"
uuid = "864edb3b-99cc-5e75-8d2d-829cb0a9cfe8"
version = "0.18.20"

[[deps.DataValueInterfaces]]
git-tree-sha1 = "bfc1187b79289637fa0ef6d4436ebdfe6905cbd6"
uuid = "e2d170a0-9d28-54be-80f0-106bbe20a464"
version = "1.0.0"

[[deps.Dates]]
deps = ["Printf"]
uuid = "ade2ca70-3891-5945-98fb-dc099432e06a"
version = "1.11.0"

[[deps.Dbus_jll]]
deps = ["Artifacts", "Expat_jll", "JLLWrappers", "Libdl"]
git-tree-sha1 = "fc173b380865f70627d7dd1190dc2fce6cc105af"
uuid = "ee1fde0b-3d02-5ea6-8484-8dfef6360eab"
version = "1.14.10+0"

[[deps.DelimitedFiles]]
deps = ["Mmap"]
git-tree-sha1 = "9e2f36d3c96a820c678f2f1f1782582fcf685bae"
uuid = "8bb1440f-4735-579b-a4ab-409b98df4dab"
version = "1.9.1"

[[deps.DiffResults]]
deps = ["StaticArraysCore"]
git-tree-sha1 = "782dd5f4561f5d267313f23853baaaa4c52ea621"
uuid = "163ba53b-c6d8-5494-b064-1a9d43ac40c5"
version = "1.1.0"

[[deps.DiffRules]]
deps = ["IrrationalConstants", "LogExpFunctions", "NaNMath", "Random", "SpecialFunctions"]
git-tree-sha1 = "23163d55f885173722d1e4cf0f6110cdbaf7e272"
uuid = "b552c78f-8df3-52c6-915a-8e097449b14b"
version = "1.15.1"

[[deps.Distributed]]
deps = ["Random", "Serialization", "Sockets"]
uuid = "8ba89e20-285c-5b6f-9357-94700520ee1b"
version = "1.11.0"

[[deps.Distributions]]
deps = ["AliasTables", "FillArrays", "LinearAlgebra", "PDMats", "Printf", "QuadGK", "Random", "SpecialFunctions", "Statistics", "StatsAPI", "StatsBase", "StatsFuns"]
git-tree-sha1 = "7901a6117656e29fa2c74a58adb682f380922c47"
uuid = "31c24e10-a181-5473-b8eb-7969acd0382f"
version = "0.25.116"

    [deps.Distributions.extensions]
    DistributionsChainRulesCoreExt = "ChainRulesCore"
    DistributionsDensityInterfaceExt = "DensityInterface"
    DistributionsTestExt = "Test"

    [deps.Distributions.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    DensityInterface = "b429d917-457f-4dbc-8f4c-0cc954292b1d"
    Test = "8dfed614-e22c-5e08-85e1-65c5234f0b40"

[[deps.DocStringExtensions]]
deps = ["LibGit2"]
git-tree-sha1 = "2fb1e02f2b635d0845df5d7c167fec4dd739b00d"
uuid = "ffbed154-4ef7-542d-bbb7-c09d3a79fcae"
version = "0.9.3"

[[deps.Downloads]]
deps = ["ArgTools", "FileWatching", "LibCURL", "NetworkOptions"]
uuid = "f43a241f-c20a-4ad4-852c-f6b1247861c6"
version = "1.6.0"

[[deps.EnumX]]
git-tree-sha1 = "bdb1942cd4c45e3c678fd11569d5cccd80976237"
uuid = "4e289a0a-7415-4d19-859d-a7e5c4648b56"
version = "1.0.4"

[[deps.EpollShim_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "8a4be429317c42cfae6a7fc03c31bad1970c310d"
uuid = "2702e6a9-849d-5ed8-8c21-79e8b8f9ee43"
version = "0.0.20230411+1"

[[deps.ErrorfreeArithmetic]]
git-tree-sha1 = "d6863c556f1142a061532e79f611aa46be201686"
uuid = "90fa49ef-747e-5e6f-a989-263ba693cf1a"
version = "0.5.2"

[[deps.ExceptionUnwrapping]]
deps = ["Test"]
git-tree-sha1 = "d36f682e590a83d63d1c7dbd287573764682d12a"
uuid = "460bff9d-24e4-43bc-9d9f-a8973cb893f4"
version = "0.1.11"

[[deps.Expat_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "e51db81749b0777b2147fbe7b783ee79045b8e99"
uuid = "2e619515-83b5-522b-bb60-26c02a35a201"
version = "2.6.4+3"

[[deps.ExprTools]]
git-tree-sha1 = "27415f162e6028e81c72b82ef756bf321213b6ec"
uuid = "e2ba6199-217a-4e67-a87a-7c52f15ade04"
version = "0.1.10"

[[deps.ExpressionExplorer]]
git-tree-sha1 = "71d0768dd78ad62d3582091bf338d98af8bbda67"
uuid = "21656369-7473-754a-2065-74616d696c43"
version = "1.1.1"

[[deps.ExproniconLite]]
git-tree-sha1 = "4c9ed87a6b3cd90acf24c556f2119533435ded38"
uuid = "55351af7-c7e9-48d6-89ff-24e801d99491"
version = "0.10.13"

[[deps.FFMPEG]]
deps = ["FFMPEG_jll"]
git-tree-sha1 = "53ebe7511fa11d33bec688a9178fac4e49eeee00"
uuid = "c87230d0-a227-11e9-1b43-d7ebe4e7570a"
version = "0.4.2"

[[deps.FFMPEG_jll]]
deps = ["Artifacts", "Bzip2_jll", "FreeType2_jll", "FriBidi_jll", "JLLWrappers", "LAME_jll", "Libdl", "Ogg_jll", "OpenSSL_jll", "Opus_jll", "PCRE2_jll", "Zlib_jll", "libaom_jll", "libass_jll", "libfdk_aac_jll", "libvorbis_jll", "x264_jll", "x265_jll"]
git-tree-sha1 = "466d45dc38e15794ec7d5d63ec03d776a9aff36e"
uuid = "b22a6f82-2f65-5046-a5b2-351ab43fb4e5"
version = "4.4.4+1"

[[deps.FastRounding]]
deps = ["ErrorfreeArithmetic", "LinearAlgebra"]
git-tree-sha1 = "6344aa18f654196be82e62816935225b3b9abe44"
uuid = "fa42c844-2597-5d31-933b-ebd51ab2693f"
version = "0.3.1"

[[deps.FileWatching]]
uuid = "7b1f6079-737a-58dc-b8bc-7a2ca5c1b5ee"
version = "1.11.0"

[[deps.FillArrays]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "6a70198746448456524cb442b8af316927ff3e1a"
uuid = "1a297f60-69ca-5386-bcde-b61e274b549b"
version = "1.13.0"
weakdeps = ["PDMats", "SparseArrays", "Statistics"]

    [deps.FillArrays.extensions]
    FillArraysPDMatsExt = "PDMats"
    FillArraysSparseArraysExt = "SparseArrays"
    FillArraysStatisticsExt = "Statistics"

[[deps.FiniteDiff]]
deps = ["ArrayInterface", "LinearAlgebra", "Setfield"]
git-tree-sha1 = "84e3a47db33be7248daa6274b287507dd6ff84e8"
uuid = "6a86dc24-6348-571c-b903-95158fe2bd41"
version = "2.26.2"

    [deps.FiniteDiff.extensions]
    FiniteDiffBandedMatricesExt = "BandedMatrices"
    FiniteDiffBlockBandedMatricesExt = "BlockBandedMatrices"
    FiniteDiffSparseArraysExt = "SparseArrays"
    FiniteDiffStaticArraysExt = "StaticArrays"

    [deps.FiniteDiff.weakdeps]
    BandedMatrices = "aae01518-5342-5314-be14-df237901396f"
    BlockBandedMatrices = "ffab5731-97b5-5995-9138-79e8c1846df0"
    SparseArrays = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
    StaticArrays = "90137ffa-7385-5640-81b9-e52037218182"

[[deps.FixedPointNumbers]]
deps = ["Statistics"]
git-tree-sha1 = "05882d6995ae5c12bb5f36dd2ed3f61c98cbb172"
uuid = "53c48c17-4a7d-5ca2-90c5-79b7896eea93"
version = "0.8.5"

[[deps.Fontconfig_jll]]
deps = ["Artifacts", "Bzip2_jll", "Expat_jll", "FreeType2_jll", "JLLWrappers", "Libdl", "Libuuid_jll", "Zlib_jll"]
git-tree-sha1 = "21fac3c77d7b5a9fc03b0ec503aa1a6392c34d2b"
uuid = "a3f928ae-7b40-5064-980b-68af3947d34b"
version = "2.15.0+0"

[[deps.Format]]
git-tree-sha1 = "9c68794ef81b08086aeb32eeaf33531668d5f5fc"
uuid = "1fa38f19-a742-5d3f-a2b9-30dd87b9d5f8"
version = "1.3.7"

[[deps.ForwardDiff]]
deps = ["CommonSubexpressions", "DiffResults", "DiffRules", "LinearAlgebra", "LogExpFunctions", "NaNMath", "Preferences", "Printf", "Random", "SpecialFunctions"]
git-tree-sha1 = "a2df1b776752e3f344e5116c06d75a10436ab853"
uuid = "f6369f11-7733-5829-9624-2563aa707210"
version = "0.10.38"
weakdeps = ["StaticArrays"]

    [deps.ForwardDiff.extensions]
    ForwardDiffStaticArraysExt = "StaticArrays"

[[deps.FreeType2_jll]]
deps = ["Artifacts", "Bzip2_jll", "JLLWrappers", "Libdl", "Zlib_jll"]
git-tree-sha1 = "786e968a8d2fb167f2e4880baba62e0e26bd8e4e"
uuid = "d7e528f0-a631-5988-bf34-fe36492bcfd7"
version = "2.13.3+1"

[[deps.FriBidi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "846f7026a9decf3679419122b49f8a1fdb48d2d5"
uuid = "559328eb-81f9-559d-9380-de523a88c83c"
version = "1.0.16+0"

[[deps.FunctionWrappers]]
git-tree-sha1 = "d62485945ce5ae9c0c48f124a84998d755bae00e"
uuid = "069b7b12-0de2-55c6-9aab-29f3d0a68a2e"
version = "1.1.3"

[[deps.Future]]
deps = ["Random"]
uuid = "9fa8497b-333b-5362-9e8d-4d0656e87820"
version = "1.11.0"

[[deps.FuzzyCompletions]]
deps = ["REPL"]
git-tree-sha1 = "be713866335f48cfb1285bff2d0cbb8304c1701c"
uuid = "fb4132e2-a121-4a70-b8a1-d5b831dcdcc2"
version = "0.5.5"

[[deps.GLFW_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libglvnd_jll", "Xorg_libXcursor_jll", "Xorg_libXi_jll", "Xorg_libXinerama_jll", "Xorg_libXrandr_jll", "libdecor_jll", "xkbcommon_jll"]
git-tree-sha1 = "fcb0584ff34e25155876418979d4c8971243bb89"
uuid = "0656b61e-2033-5cc2-a64a-77c0f6c09b89"
version = "3.4.0+2"

[[deps.GLPK]]
deps = ["GLPK_jll", "MathOptInterface"]
git-tree-sha1 = "1d706bd23e5d2d407bfd369499ee6f96afb0c3ad"
uuid = "60bf3e95-4087-53dc-ae20-288a0d20c6a6"
version = "1.2.1"

[[deps.GLPK_jll]]
deps = ["Artifacts", "GMP_jll", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "6aa6294ba949ccfc380463bf50ff988b46de5bc7"
uuid = "e8aa6df9-e6ca-548a-97ff-1f85fc5b8b98"
version = "5.0.1+1"

[[deps.GMP_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "781609d7-10c4-51f6-84f2-b8444358ff6d"
version = "6.3.0+0"

[[deps.GPUArrays]]
deps = ["Adapt", "GPUArraysCore", "KernelAbstractions", "LLVM", "LinearAlgebra", "Printf", "Random", "Reexport", "ScopedValues", "Serialization", "Statistics"]
git-tree-sha1 = "0ef97e93edced3d0e713f4cfd031cc9020e022b0"
uuid = "0c68f7d7-f131-5f86-a1c3-88cf8149b2d7"
version = "11.2.1"

[[deps.GPUArraysCore]]
deps = ["Adapt"]
git-tree-sha1 = "83cf05ab16a73219e5f6bd1bdfa9848fa24ac627"
uuid = "46192b85-c4d5-4398-a991-12ede77f4527"
version = "0.2.0"

[[deps.GR]]
deps = ["Artifacts", "Base64", "DelimitedFiles", "Downloads", "GR_jll", "HTTP", "JSON", "Libdl", "LinearAlgebra", "Preferences", "Printf", "Qt6Wayland_jll", "Random", "Serialization", "Sockets", "TOML", "Tar", "Test", "p7zip_jll"]
git-tree-sha1 = "424c8f76017e39fdfcdbb5935a8e6742244959e8"
uuid = "28b8d3ca-fb5f-59d9-8090-bfdbd6d07a71"
version = "0.73.10"

[[deps.GR_jll]]
deps = ["Artifacts", "Bzip2_jll", "Cairo_jll", "FFMPEG_jll", "Fontconfig_jll", "FreeType2_jll", "GLFW_jll", "JLLWrappers", "JpegTurbo_jll", "Libdl", "Libtiff_jll", "Pixman_jll", "Qt6Base_jll", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "b90934c8cb33920a8dc66736471dc3961b42ec9f"
uuid = "d2c73de3-f751-5644-a686-071e5b155ba9"
version = "0.73.10+0"

[[deps.Gettext_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Libiconv_jll", "Pkg", "XML2_jll"]
git-tree-sha1 = "9b02998aba7bf074d14de89f9d37ca24a1a0b046"
uuid = "78b55507-aeef-58d4-861c-77aaff3498b1"
version = "0.21.0+0"

[[deps.Glib_jll]]
deps = ["Artifacts", "Gettext_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Libiconv_jll", "Libmount_jll", "PCRE2_jll", "Zlib_jll"]
git-tree-sha1 = "b0036b392358c80d2d2124746c2bf3d48d457938"
uuid = "7746bdde-850d-59dc-9ae8-88ece973131d"
version = "2.82.4+0"

[[deps.Graphite2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "01979f9b37367603e2848ea225918a3b3861b606"
uuid = "3b182d85-2403-5c21-9c21-1e1f0cc25472"
version = "1.3.14+1"

[[deps.GridInterpolations]]
deps = ["LinearAlgebra", "Printf", "StaticArrays"]
git-tree-sha1 = "e64e58d732c7c1f32575e2b057c0fb0f7f52e244"
uuid = "bb4c363b-b914-514b-8517-4eb369bc008a"
version = "1.2.1"

[[deps.Grisu]]
git-tree-sha1 = "53bb909d1151e57e2484c3d1b53e19552b887fb2"
uuid = "42e2da0e-8278-4e71-bc24-59509adca0fe"
version = "1.0.2"

[[deps.HTTP]]
deps = ["Base64", "CodecZlib", "ConcurrentUtilities", "Dates", "ExceptionUnwrapping", "Logging", "LoggingExtras", "MbedTLS", "NetworkOptions", "OpenSSL", "PrecompileTools", "Random", "SimpleBufferStream", "Sockets", "URIs", "UUIDs"]
git-tree-sha1 = "c67b33b085f6e2faf8bf79a61962e7339a81129c"
uuid = "cd3eb016-35fb-5094-929b-558a96fad6f3"
version = "1.10.15"

[[deps.HarfBuzz_jll]]
deps = ["Artifacts", "Cairo_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "Graphite2_jll", "JLLWrappers", "Libdl", "Libffi_jll"]
git-tree-sha1 = "55c53be97790242c29031e5cd45e8ac296dadda3"
uuid = "2e76f6c2-a576-52d4-95c1-20adfe4de566"
version = "8.5.0+0"

[[deps.HashArrayMappedTries]]
git-tree-sha1 = "2eaa69a7cab70a52b9687c8bf950a5a93ec895ae"
uuid = "076d061b-32b6-4027-95e0-9a2c6f6d7e74"
version = "0.2.0"

[[deps.HypergeometricFunctions]]
deps = ["LinearAlgebra", "OpenLibm_jll", "SpecialFunctions"]
git-tree-sha1 = "b1c2585431c382e3fe5805874bda6aea90a95de9"
uuid = "34004b35-14d8-5ef3-9330-4cdb6864b03a"
version = "0.3.25"

[[deps.Hyperscript]]
deps = ["Test"]
git-tree-sha1 = "179267cfa5e712760cd43dcae385d7ea90cc25a4"
uuid = "47d2ed2b-36de-50cf-bf87-49c2cf4b8b91"
version = "0.0.5"

[[deps.HypertextLiteral]]
deps = ["Tricks"]
git-tree-sha1 = "7134810b1afce04bbc1045ca1985fbe81ce17653"
uuid = "ac1192a8-f4b3-4bfe-ba22-af5b92cd3ab2"
version = "0.9.5"

[[deps.IOCapture]]
deps = ["Logging", "Random"]
git-tree-sha1 = "b6d6bfdd7ce25b0f9b2f6b3dd56b2673a66c8770"
uuid = "b5f81e59-6552-4d32-b1f0-c071b021bf89"
version = "0.2.5"

[[deps.IRTools]]
deps = ["InteractiveUtils", "MacroTools"]
git-tree-sha1 = "950c3717af761bc3ff906c2e8e52bd83390b6ec2"
uuid = "7869d1d1-7146-5819-86e3-90919afe41df"
version = "0.4.14"

[[deps.InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"
version = "1.11.0"

[[deps.IntervalArithmetic]]
deps = ["CRlibm", "EnumX", "FastRounding", "LinearAlgebra", "Markdown", "Random", "RecipesBase", "RoundingEmulator", "SetRounding", "StaticArrays"]
git-tree-sha1 = "f59e639916283c1d2e106d2b00910b50f4dab76c"
uuid = "d1acc4aa-44c8-5952-acd4-ba5d80a2a253"
version = "0.21.2"

[[deps.IrrationalConstants]]
git-tree-sha1 = "630b497eafcc20001bba38a4651b327dcfc491d2"
uuid = "92d709cd-6900-40b7-9082-c6be49f344b6"
version = "0.2.2"

[[deps.IteratorInterfaceExtensions]]
git-tree-sha1 = "a3f24677c21f5bbe9d2a714f95dcd58337fb2856"
uuid = "82899510-4779-5014-852e-03e436cf321d"
version = "1.0.0"

[[deps.JLFzf]]
deps = ["Pipe", "REPL", "Random", "fzf_jll"]
git-tree-sha1 = "71b48d857e86bf7a1838c4736545699974ce79a2"
uuid = "1019f520-868f-41f5-a6de-eb00f4b6a39c"
version = "0.1.9"

[[deps.JLLWrappers]]
deps = ["Artifacts", "Preferences"]
git-tree-sha1 = "a007feb38b422fbdab534406aeca1b86823cb4d6"
uuid = "692b3bcd-3c85-4b1f-b108-f13ce0eb3210"
version = "1.7.0"

[[deps.JSON]]
deps = ["Dates", "Mmap", "Parsers", "Unicode"]
git-tree-sha1 = "31e996f0a15c7b280ba9f76636b3ff9e2ae58c9a"
uuid = "682c06a0-de6a-54ab-a142-c8b1cf79cde6"
version = "0.21.4"

[[deps.JpegTurbo_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "eac1206917768cb54957c65a615460d87b455fc1"
uuid = "aacddb02-875f-59d6-b918-886e6ef4fbf8"
version = "3.1.1+0"

[[deps.JuMP]]
deps = ["LinearAlgebra", "MacroTools", "MathOptInterface", "MutableArithmetics", "OrderedCollections", "PrecompileTools", "Printf", "SparseArrays"]
git-tree-sha1 = "02b6e65736debc1f47b40b0f7d5dfa0217ee1f09"
uuid = "4076af6c-e467-56ae-b986-b466b2749572"
version = "1.23.6"

    [deps.JuMP.extensions]
    JuMPDimensionalDataExt = "DimensionalData"

    [deps.JuMP.weakdeps]
    DimensionalData = "0703355e-b756-11e9-17c0-8b28908087d0"

[[deps.KernelAbstractions]]
deps = ["Adapt", "Atomix", "InteractiveUtils", "MacroTools", "PrecompileTools", "Requires", "StaticArrays", "UUIDs"]
git-tree-sha1 = "b9a838cd3028785ac23822cded5126b3da394d1a"
uuid = "63c18a36-062a-441e-b654-da1e3ab1ce7c"
version = "0.9.31"

    [deps.KernelAbstractions.extensions]
    EnzymeExt = "EnzymeCore"
    LinearAlgebraExt = "LinearAlgebra"
    SparseArraysExt = "SparseArrays"

    [deps.KernelAbstractions.weakdeps]
    EnzymeCore = "f151be2c-9106-41f4-ab19-57ee4f262869"
    LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
    SparseArrays = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"

[[deps.LAME_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "170b660facf5df5de098d866564877e119141cbd"
uuid = "c1c5ebd0-6772-5130-a774-d5fcae4a789d"
version = "3.100.2+0"

[[deps.LERC_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "aaafe88dccbd957a8d82f7d05be9b69172e0cee3"
uuid = "88015f11-f218-50d7-93a8-a6af411a945d"
version = "4.0.1+0"

[[deps.LLVM]]
deps = ["CEnum", "LLVMExtra_jll", "Libdl", "Preferences", "Printf", "Unicode"]
git-tree-sha1 = "d422dfd9707bec6617335dc2ea3c5172a87d5908"
uuid = "929cbde3-209d-540e-8aea-75f648917ca0"
version = "9.1.3"

    [deps.LLVM.extensions]
    BFloat16sExt = "BFloat16s"

    [deps.LLVM.weakdeps]
    BFloat16s = "ab4f0b2a-ad5b-11e8-123f-65d77653426b"

[[deps.LLVMExtra_jll]]
deps = ["Artifacts", "JLLWrappers", "LazyArtifacts", "Libdl", "TOML"]
git-tree-sha1 = "05a8bd5a42309a9ec82f700876903abce1017dd3"
uuid = "dad2f222-ce93-54a1-a47d-0025e8a3acab"
version = "0.0.34+0"

[[deps.LLVMOpenMP_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "78211fb6cbc872f77cad3fc0b6cf647d923f4929"
uuid = "1d63c593-3942-5779-bab2-d838dc0a180e"
version = "18.1.7+0"

[[deps.LZO_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "1c602b1127f4751facb671441ca72715cc95938a"
uuid = "dd4b983a-f0e5-5f8d-a1b7-129d4a5fb1ac"
version = "2.10.3+0"

[[deps.LaTeXStrings]]
git-tree-sha1 = "dda21b8cbd6a6c40d9d02a73230f9d70fed6918c"
uuid = "b964fa9f-0449-5b57-a5c2-d3ea65f4040f"
version = "1.4.0"

[[deps.Latexify]]
deps = ["Format", "InteractiveUtils", "LaTeXStrings", "MacroTools", "Markdown", "OrderedCollections", "Requires"]
git-tree-sha1 = "ce5f5621cac23a86011836badfedf664a612cee4"
uuid = "23fbe1c1-3f47-55db-b15f-69d7ec21a316"
version = "0.16.5"

    [deps.Latexify.extensions]
    DataFramesExt = "DataFrames"
    SparseArraysExt = "SparseArrays"
    SymEngineExt = "SymEngine"

    [deps.Latexify.weakdeps]
    DataFrames = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
    SparseArrays = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
    SymEngine = "123dc426-2d89-5057-bbad-38513e3affd8"

[[deps.LazilyInitializedFields]]
git-tree-sha1 = "0f2da712350b020bc3957f269c9caad516383ee0"
uuid = "0e77f7df-68c5-4e49-93ce-4cd80f5598bf"
version = "1.3.0"

[[deps.LazyArtifacts]]
deps = ["Artifacts", "Pkg"]
uuid = "4af54fe1-eca0-43a8-85a7-787d91b784e3"
version = "1.11.0"

[[deps.LazySets]]
deps = ["Distributed", "GLPK", "IntervalArithmetic", "JuMP", "LinearAlgebra", "Random", "ReachabilityBase", "RecipesBase", "Reexport", "Requires", "SharedArrays", "SparseArrays", "StaticArraysCore"]
git-tree-sha1 = "ae9b6a027c694b9e0bab91fc25d0b2808f1bf755"
uuid = "b4f0291d-fe17-52bc-9479-3d1a343d9043"
version = "3.0.0"

[[deps.LibCURL]]
deps = ["LibCURL_jll", "MozillaCACerts_jll"]
uuid = "b27032c2-a3e7-50c8-80cd-2d36dbcbfd21"
version = "0.6.4"

[[deps.LibCURL_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "MbedTLS_jll", "Zlib_jll", "nghttp2_jll"]
uuid = "deac9b47-8bc7-5906-a0fe-35ac56dc84c0"
version = "8.6.0+0"

[[deps.LibGit2]]
deps = ["Base64", "LibGit2_jll", "NetworkOptions", "Printf", "SHA"]
uuid = "76f85450-5226-5b5a-8eaa-529ad045b433"
version = "1.11.0"

[[deps.LibGit2_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "MbedTLS_jll"]
uuid = "e37daf67-58a4-590a-8e99-b0245dd2ffc5"
version = "1.7.2+0"

[[deps.LibSSH2_jll]]
deps = ["Artifacts", "Libdl", "MbedTLS_jll"]
uuid = "29816b5a-b9ab-546f-933c-edad1886dfa8"
version = "1.11.0+1"

[[deps.Libdl]]
uuid = "8f399da3-3557-5675-b5ff-fb832c97cbdb"
version = "1.11.0"

[[deps.Libffi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "27ecae93dd25ee0909666e6835051dd684cc035e"
uuid = "e9f186c6-92d2-5b65-8a66-fee21dc1b490"
version = "3.2.2+2"

[[deps.Libgcrypt_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libgpg_error_jll"]
git-tree-sha1 = "8be878062e0ffa2c3f67bb58a595375eda5de80b"
uuid = "d4300ac3-e22c-5743-9152-c294e39db1e4"
version = "1.11.0+0"

[[deps.Libglvnd_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libX11_jll", "Xorg_libXext_jll"]
git-tree-sha1 = "ff3b4b9d35de638936a525ecd36e86a8bb919d11"
uuid = "7e76a0d4-f3c7-5321-8279-8d96eeed0f29"
version = "1.7.0+0"

[[deps.Libgpg_error_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "df37206100d39f79b3376afb6b9cee4970041c61"
uuid = "7add5ba3-2f88-524e-9cd5-f83b8a55f7b8"
version = "1.51.1+0"

[[deps.Libiconv_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "be484f5c92fad0bd8acfef35fe017900b0b73809"
uuid = "94ce4f54-9a6c-5748-9c1c-f9c7231a4531"
version = "1.18.0+0"

[[deps.Libmount_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "89211ea35d9df5831fca5d33552c02bd33878419"
uuid = "4b2f31a3-9ecc-558c-b454-b3730dcb73e9"
version = "2.40.3+0"

[[deps.Libtiff_jll]]
deps = ["Artifacts", "JLLWrappers", "JpegTurbo_jll", "LERC_jll", "Libdl", "XZ_jll", "Zlib_jll", "Zstd_jll"]
git-tree-sha1 = "4ab7581296671007fc33f07a721631b8855f4b1d"
uuid = "89763e89-9b03-5906-acba-b20f662cd828"
version = "4.7.1+0"

[[deps.Libuuid_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "e888ad02ce716b319e6bdb985d2ef300e7089889"
uuid = "38a345b3-de98-5d2b-a5d3-14cd9215e700"
version = "2.40.3+0"

[[deps.LineSearches]]
deps = ["LinearAlgebra", "NLSolversBase", "NaNMath", "Parameters", "Printf"]
git-tree-sha1 = "e4c3be53733db1051cc15ecf573b1042b3a712a1"
uuid = "d3d80556-e9d4-5f37-9878-2ab0fcc64255"
version = "7.3.0"

[[deps.LinearAlgebra]]
deps = ["Libdl", "OpenBLAS_jll", "libblastrampoline_jll"]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
version = "1.11.0"

[[deps.LogExpFunctions]]
deps = ["DocStringExtensions", "IrrationalConstants", "LinearAlgebra"]
git-tree-sha1 = "13ca9e2586b89836fd20cccf56e57e2b9ae7f38f"
uuid = "2ab3a3ac-af41-5b50-aa03-7779005ae688"
version = "0.3.29"

    [deps.LogExpFunctions.extensions]
    LogExpFunctionsChainRulesCoreExt = "ChainRulesCore"
    LogExpFunctionsChangesOfVariablesExt = "ChangesOfVariables"
    LogExpFunctionsInverseFunctionsExt = "InverseFunctions"

    [deps.LogExpFunctions.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    ChangesOfVariables = "9e997f8a-9a97-42d5-a9f1-ce6bfc15e2c0"
    InverseFunctions = "3587e190-3f89-42d0-90ee-14403ec27112"

[[deps.Logging]]
uuid = "56ddb016-857b-54e1-b83d-db4d58db5568"
version = "1.11.0"

[[deps.LoggingExtras]]
deps = ["Dates", "Logging"]
git-tree-sha1 = "f02b56007b064fbfddb4c9cd60161b6dd0f40df3"
uuid = "e6f89c97-d47a-5376-807f-9c37f3926c36"
version = "1.1.0"

[[deps.MIMEs]]
git-tree-sha1 = "65f28ad4b594aebe22157d6fac869786a255b7eb"
uuid = "6c6e2e6c-3030-632d-7369-2d6c69616d65"
version = "0.1.4"

[[deps.MacroTools]]
git-tree-sha1 = "72aebe0b5051e5143a079a4685a46da330a40472"
uuid = "1914dd2f-81c6-5fcd-8719-6d5c9610ff09"
version = "0.5.15"

[[deps.Malt]]
deps = ["Distributed", "Logging", "RelocatableFolders", "Serialization", "Sockets"]
git-tree-sha1 = "02a728ada9d6caae583a0f87c1dd3844f99ec3fd"
uuid = "36869731-bdee-424d-aa32-cab38c994e3b"
version = "1.1.2"

[[deps.Markdown]]
deps = ["Base64"]
uuid = "d6f4376e-aef5-505a-96c1-9c027394607a"
version = "1.11.0"

[[deps.MathOptInterface]]
deps = ["BenchmarkTools", "CodecBzip2", "CodecZlib", "DataStructures", "ForwardDiff", "JSON", "LinearAlgebra", "MutableArithmetics", "NaNMath", "OrderedCollections", "PrecompileTools", "Printf", "SparseArrays", "SpecialFunctions", "Test", "Unicode"]
git-tree-sha1 = "2974c2d3577c8a4cff150cd03e589d637a864276"
uuid = "b8f27783-ece8-5eb3-8dc8-9495eed66fee"
version = "1.35.0"

[[deps.MbedTLS]]
deps = ["Dates", "MbedTLS_jll", "MozillaCACerts_jll", "NetworkOptions", "Random", "Sockets"]
git-tree-sha1 = "c067a280ddc25f196b5e7df3877c6b226d390aaf"
uuid = "739be429-bea8-5141-9913-cc70e7f3736d"
version = "1.1.9"

[[deps.MbedTLS_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "c8ffd9c3-330d-5841-b78e-0817d7145fa1"
version = "2.28.6+0"

[[deps.Measures]]
git-tree-sha1 = "c13304c81eec1ed3af7fc20e75fb6b26092a1102"
uuid = "442fdcdd-2543-5da2-b0f3-8c86c306513e"
version = "0.3.2"

[[deps.Missings]]
deps = ["DataAPI"]
git-tree-sha1 = "ec4f7fbeab05d7747bdf98eb74d130a2a2ed298d"
uuid = "e1d29d7a-bbdc-5cf2-9ac0-f12de2c33e28"
version = "1.2.0"

[[deps.Mmap]]
uuid = "a63ad114-7e13-5084-954f-fe012c677804"
version = "1.11.0"

[[deps.MozillaCACerts_jll]]
uuid = "14a3606d-f60d-562e-9121-12d972cd8159"
version = "2023.12.12"

[[deps.MsgPack]]
deps = ["Serialization"]
git-tree-sha1 = "f5db02ae992c260e4826fe78c942954b48e1d9c2"
uuid = "99f44e22-a591-53d1-9472-aa23ef4bd671"
version = "1.2.1"

[[deps.MutableArithmetics]]
deps = ["LinearAlgebra", "SparseArrays", "Test"]
git-tree-sha1 = "a2710df6b0931f987530f59427441b21245d8f5e"
uuid = "d8a4904e-b15c-11e9-3269-09a3773c0cb0"
version = "1.6.0"

[[deps.NLSolversBase]]
deps = ["DiffResults", "Distributed", "FiniteDiff", "ForwardDiff"]
git-tree-sha1 = "a0b464d183da839699f4c79e7606d9d186ec172c"
uuid = "d41bc354-129a-5804-8e4c-c37616107c6c"
version = "7.8.3"

[[deps.NaNMath]]
deps = ["OpenLibm_jll"]
git-tree-sha1 = "030ea22804ef91648f29b7ad3fc15fa49d0e6e71"
uuid = "77ba4419-2d1f-58cd-9bb1-8ffee604a2e3"
version = "1.0.3"

[[deps.NetworkOptions]]
uuid = "ca575930-c2e3-43a9-ace4-1e988b2c1908"
version = "1.2.0"

[[deps.Ogg_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "887579a3eb005446d514ab7aeac5d1d027658b8f"
uuid = "e7412a2a-1a6e-54c0-be00-318e2571c051"
version = "1.3.5+1"

[[deps.OpenBLAS_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "4536629a-c528-5b80-bd46-f80d51c5b363"
version = "0.3.27+1"

[[deps.OpenLibm_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "05823500-19ac-5b8b-9628-191a04bc5112"
version = "0.8.1+2"

[[deps.OpenSSL]]
deps = ["BitFlags", "Dates", "MozillaCACerts_jll", "OpenSSL_jll", "Sockets"]
git-tree-sha1 = "38cb508d080d21dc1128f7fb04f20387ed4c0af4"
uuid = "4d8831e6-92b7-49fb-bdf8-b643e874388c"
version = "1.4.3"

[[deps.OpenSSL_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "7493f61f55a6cce7325f197443aa80d32554ba10"
uuid = "458c3c95-2e84-50aa-8efc-19380b2a3a95"
version = "3.0.15+3"

[[deps.OpenSpecFun_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl"]
git-tree-sha1 = "1346c9208249809840c91b26703912dff463d335"
uuid = "efe28fd5-8261-553b-a9e1-b2916fc3738e"
version = "0.5.6+0"

[[deps.Optim]]
deps = ["Compat", "FillArrays", "ForwardDiff", "LineSearches", "LinearAlgebra", "NLSolversBase", "NaNMath", "Parameters", "PositiveFactorizations", "Printf", "SparseArrays", "StatsBase"]
git-tree-sha1 = "ab7edad78cdef22099f43c54ef77ac63c2c9cc64"
uuid = "429524aa-4258-5aef-a3af-852621145aeb"
version = "1.10.0"
weakdeps = ["MathOptInterface"]

    [deps.Optim.extensions]
    OptimMOIExt = "MathOptInterface"

[[deps.Opus_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "6703a85cb3781bd5909d48730a67205f3f31a575"
uuid = "91d4177d-7536-5919-b921-800302f37372"
version = "1.3.3+0"

[[deps.OrderedCollections]]
git-tree-sha1 = "12f1439c4f986bb868acda6ea33ebc78e19b95ad"
uuid = "bac558e1-5e72-5ebc-8fee-abe8a469f55d"
version = "1.7.0"

[[deps.PCRE2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "efcefdf7-47ab-520b-bdef-62a2eaa19f15"
version = "10.42.0+1"

[[deps.PDMats]]
deps = ["LinearAlgebra", "SparseArrays", "SuiteSparse"]
git-tree-sha1 = "949347156c25054de2db3b166c52ac4728cbad65"
uuid = "90014a1f-27ba-587c-ab20-58faa44d9150"
version = "0.11.31"

[[deps.Pango_jll]]
deps = ["Artifacts", "Cairo_jll", "Fontconfig_jll", "FreeType2_jll", "FriBidi_jll", "Glib_jll", "HarfBuzz_jll", "JLLWrappers", "Libdl"]
git-tree-sha1 = "ed6834e95bd326c52d5675b4181386dfbe885afb"
uuid = "36c8627f-9965-5494-a995-c6b170f724f3"
version = "1.55.5+0"

[[deps.Parameters]]
deps = ["OrderedCollections", "UnPack"]
git-tree-sha1 = "34c0e9ad262e5f7fc75b10a9952ca7692cfc5fbe"
uuid = "d96e819e-fc66-5662-9728-84c9c7592b0a"
version = "0.12.3"

[[deps.Parsers]]
deps = ["Dates", "PrecompileTools", "UUIDs"]
git-tree-sha1 = "8489905bcdbcfac64d1daa51ca07c0d8f0283821"
uuid = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"
version = "2.8.1"

[[deps.Pipe]]
git-tree-sha1 = "6842804e7867b115ca9de748a0cf6b364523c16d"
uuid = "b98c9c47-44ae-5843-9183-064241ee97a0"
version = "1.3.0"

[[deps.Pixman_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "LLVMOpenMP_jll", "Libdl"]
git-tree-sha1 = "35621f10a7531bc8fa58f74610b1bfb70a3cfc6b"
uuid = "30392449-352a-5448-841d-b1acce4e97dc"
version = "0.43.4+0"

[[deps.Pkg]]
deps = ["Artifacts", "Dates", "Downloads", "FileWatching", "LibGit2", "Libdl", "Logging", "Markdown", "Printf", "Random", "SHA", "TOML", "Tar", "UUIDs", "p7zip_jll"]
uuid = "44cfe95a-1eb2-52ea-b672-e2afdf69b78f"
version = "1.11.0"
weakdeps = ["REPL"]

    [deps.Pkg.extensions]
    REPLExt = "REPL"

[[deps.PlotThemes]]
deps = ["PlotUtils", "Statistics"]
git-tree-sha1 = "41031ef3a1be6f5bbbf3e8073f210556daeae5ca"
uuid = "ccf2f8ad-2431-5c83-bf29-c5338b663b6a"
version = "3.3.0"

[[deps.PlotUtils]]
deps = ["ColorSchemes", "Colors", "Dates", "PrecompileTools", "Printf", "Random", "Reexport", "StableRNGs", "Statistics"]
git-tree-sha1 = "3ca9a356cd2e113c420f2c13bea19f8d3fb1cb18"
uuid = "995b91a9-d308-5afd-9ec6-746e21dbc043"
version = "1.4.3"

[[deps.Plots]]
deps = ["Base64", "Contour", "Dates", "Downloads", "FFMPEG", "FixedPointNumbers", "GR", "JLFzf", "JSON", "LaTeXStrings", "Latexify", "LinearAlgebra", "Measures", "NaNMath", "Pkg", "PlotThemes", "PlotUtils", "PrecompileTools", "Printf", "REPL", "Random", "RecipesBase", "RecipesPipeline", "Reexport", "RelocatableFolders", "Requires", "Scratch", "Showoff", "SparseArrays", "Statistics", "StatsBase", "TOML", "UUIDs", "UnicodeFun", "UnitfulLatexify", "Unzip"]
git-tree-sha1 = "dae01f8c2e069a683d3a6e17bbae5070ab94786f"
uuid = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
version = "1.40.9"

    [deps.Plots.extensions]
    FileIOExt = "FileIO"
    GeometryBasicsExt = "GeometryBasics"
    IJuliaExt = "IJulia"
    ImageInTerminalExt = "ImageInTerminal"
    UnitfulExt = "Unitful"

    [deps.Plots.weakdeps]
    FileIO = "5789e2e9-d7fb-5bc7-8068-2c6fae9b9549"
    GeometryBasics = "5c1252a2-5f33-56bf-86c9-59e7332b4326"
    IJulia = "7073ff75-c697-5162-941a-fcdaad2a7d2a"
    ImageInTerminal = "d8c32880-2388-543b-8c61-d9f865259254"
    Unitful = "1986cc42-f94f-5a68-af5c-568840ba703d"

[[deps.Pluto]]
deps = ["Base64", "Configurations", "Dates", "Downloads", "ExpressionExplorer", "FileWatching", "FuzzyCompletions", "HTTP", "HypertextLiteral", "InteractiveUtils", "Logging", "LoggingExtras", "MIMEs", "Malt", "Markdown", "MsgPack", "Pkg", "PlutoDependencyExplorer", "PrecompileSignatures", "PrecompileTools", "REPL", "RegistryInstances", "RelocatableFolders", "Scratch", "Sockets", "TOML", "Tables", "URIs", "UUIDs"]
git-tree-sha1 = "b5509a2e4d4c189da505b780e3f447d1e38a0350"
uuid = "c3e4b0f8-55cb-11ea-2926-15256bba5781"
version = "0.20.4"

[[deps.PlutoDependencyExplorer]]
deps = ["ExpressionExplorer", "InteractiveUtils", "Markdown"]
git-tree-sha1 = "e0864c15334d2c4bac8137ce3359f1174565e719"
uuid = "72656b73-756c-7461-726b-72656b6b696b"
version = "1.2.0"

[[deps.PlutoUI]]
deps = ["AbstractPlutoDingetjes", "Base64", "ColorTypes", "Dates", "FixedPointNumbers", "Hyperscript", "HypertextLiteral", "IOCapture", "InteractiveUtils", "JSON", "Logging", "MIMEs", "Markdown", "Random", "Reexport", "URIs", "UUIDs"]
git-tree-sha1 = "eba4810d5e6a01f612b948c9fa94f905b49087b0"
uuid = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
version = "0.7.60"

[[deps.PositiveFactorizations]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "17275485f373e6673f7e7f97051f703ed5b15b20"
uuid = "85a6dd25-e78a-55b7-8502-1745935b8125"
version = "0.2.4"

[[deps.PrecompileSignatures]]
git-tree-sha1 = "18ef344185f25ee9d51d80e179f8dad33dc48eb1"
uuid = "91cefc8d-f054-46dc-8f8c-26e11d7c5411"
version = "3.0.3"

[[deps.PrecompileTools]]
deps = ["Preferences"]
git-tree-sha1 = "5aa36f7049a63a1528fe8f7c3f2113413ffd4e1f"
uuid = "aea7be01-6a6a-4083-8856-8a6e6704d82a"
version = "1.2.1"

[[deps.Preferences]]
deps = ["TOML"]
git-tree-sha1 = "9306f6085165d270f7e3db02af26a400d580f5c6"
uuid = "21216c6a-2e73-6563-6e65-726566657250"
version = "1.4.3"

[[deps.Printf]]
deps = ["Unicode"]
uuid = "de0858da-6303-5e67-8744-51eddeeeb8d7"
version = "1.11.0"

[[deps.Profile]]
uuid = "9abbd945-dff8-562f-b5e8-e1ebf5ef1b79"
version = "1.11.0"

[[deps.ProgressLogging]]
deps = ["Logging", "SHA", "UUIDs"]
git-tree-sha1 = "80d919dee55b9c50e8d9e2da5eeafff3fe58b539"
uuid = "33c8b6b6-d38a-422a-b730-caa89a2f386c"
version = "0.1.4"

[[deps.PtrArrays]]
git-tree-sha1 = "77a42d78b6a92df47ab37e177b2deac405e1c88f"
uuid = "43287f4e-b6f4-7ad1-bb20-aadabca52c3d"
version = "1.2.1"

[[deps.Qt6Base_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Fontconfig_jll", "Glib_jll", "JLLWrappers", "Libdl", "Libglvnd_jll", "OpenSSL_jll", "Vulkan_Loader_jll", "Xorg_libSM_jll", "Xorg_libXext_jll", "Xorg_libXrender_jll", "Xorg_libxcb_jll", "Xorg_xcb_util_cursor_jll", "Xorg_xcb_util_image_jll", "Xorg_xcb_util_keysyms_jll", "Xorg_xcb_util_renderutil_jll", "Xorg_xcb_util_wm_jll", "Zlib_jll", "libinput_jll", "xkbcommon_jll"]
git-tree-sha1 = "492601870742dcd38f233b23c3ec629628c1d724"
uuid = "c0090381-4147-56d7-9ebc-da0b1113ec56"
version = "6.7.1+1"

[[deps.Qt6Declarative_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Qt6Base_jll", "Qt6ShaderTools_jll"]
git-tree-sha1 = "e5dd466bf2569fe08c91a2cc29c1003f4797ac3b"
uuid = "629bc702-f1f5-5709-abd5-49b8460ea067"
version = "6.7.1+2"

[[deps.Qt6ShaderTools_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Qt6Base_jll"]
git-tree-sha1 = "1a180aeced866700d4bebc3120ea1451201f16bc"
uuid = "ce943373-25bb-56aa-8eca-768745ed7b5a"
version = "6.7.1+1"

[[deps.Qt6Wayland_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Qt6Base_jll", "Qt6Declarative_jll"]
git-tree-sha1 = "729927532d48cf79f49070341e1d918a65aba6b0"
uuid = "e99dba38-086e-5de3-a5b1-6e4c66e897c3"
version = "6.7.1+1"

[[deps.QuadGK]]
deps = ["DataStructures", "LinearAlgebra"]
git-tree-sha1 = "cda3b045cf9ef07a08ad46731f5a3165e56cf3da"
uuid = "1fd47b50-473d-5c70-9696-f719f8f3bcdc"
version = "2.11.1"

    [deps.QuadGK.extensions]
    QuadGKEnzymeExt = "Enzyme"

    [deps.QuadGK.weakdeps]
    Enzyme = "7da242da-08ed-463a-9acd-ee780be4f1d9"

[[deps.REPL]]
deps = ["InteractiveUtils", "Markdown", "Sockets", "StyledStrings", "Unicode"]
uuid = "3fa0cd96-eef1-5676-8a61-b3b8758bbffb"
version = "1.11.0"

[[deps.Random]]
deps = ["SHA"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"
version = "1.11.0"

[[deps.ReachabilityBase]]
deps = ["ExprTools", "InteractiveUtils", "LinearAlgebra", "Random", "Requires", "SparseArrays"]
git-tree-sha1 = "d28da1989cc21fcf57611f928061de5e8f27dc5c"
uuid = "379f33d0-9447-4353-bd03-d664070e549f"
version = "0.3.3"

[[deps.RealDot]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "9f0a1b71baaf7650f4fa8a1d168c7fb6ee41f0c9"
uuid = "c1ae055f-0cd5-4b69-90a6-9a35b1a98df9"
version = "0.1.0"

[[deps.RecipesBase]]
deps = ["PrecompileTools"]
git-tree-sha1 = "5c3d09cc4f31f5fc6af001c250bf1278733100ff"
uuid = "3cdcf5f2-1ef4-517c-9805-6587b60abb01"
version = "1.3.4"

[[deps.RecipesPipeline]]
deps = ["Dates", "NaNMath", "PlotUtils", "PrecompileTools", "RecipesBase"]
git-tree-sha1 = "45cf9fd0ca5839d06ef333c8201714e888486342"
uuid = "01d81517-befc-4cb6-b9ec-a95719d0359c"
version = "0.6.12"

[[deps.Reexport]]
git-tree-sha1 = "45e428421666073eab6f2da5c9d310d99bb12f9b"
uuid = "189a3867-3050-52da-a836-e630ba90ab69"
version = "1.2.2"

[[deps.RegistryInstances]]
deps = ["LazilyInitializedFields", "Pkg", "TOML", "Tar"]
git-tree-sha1 = "ffd19052caf598b8653b99404058fce14828be51"
uuid = "2792f1a3-b283-48e8-9a74-f99dce5104f3"
version = "0.1.0"

[[deps.RelocatableFolders]]
deps = ["SHA", "Scratch"]
git-tree-sha1 = "ffdaf70d81cf6ff22c2b6e733c900c3321cab864"
uuid = "05181044-ff0b-4ac5-8273-598c1e38db00"
version = "1.0.1"

[[deps.Requires]]
deps = ["UUIDs"]
git-tree-sha1 = "838a3a4188e2ded87a4f9f184b4b0d78a1e91cb7"
uuid = "ae029012-a4dd-5104-9daa-d747884805df"
version = "1.3.0"

[[deps.ReverseDiff]]
deps = ["ChainRulesCore", "DiffResults", "DiffRules", "ForwardDiff", "FunctionWrappers", "LinearAlgebra", "LogExpFunctions", "MacroTools", "NaNMath", "Random", "SpecialFunctions", "StaticArrays", "Statistics"]
git-tree-sha1 = "cc6cd622481ea366bb9067859446a8b01d92b468"
uuid = "37e2e3b7-166d-5795-8a7a-e32c996b4267"
version = "1.15.3"

[[deps.Rmath]]
deps = ["Random", "Rmath_jll"]
git-tree-sha1 = "852bd0f55565a9e973fcfee83a84413270224dc4"
uuid = "79098fc4-a85e-5d69-aa6a-4863f24498fa"
version = "0.8.0"

[[deps.Rmath_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "58cdd8fb2201a6267e1db87ff148dd6c1dbd8ad8"
uuid = "f50d1b31-88e8-58de-be2c-1cc44531875f"
version = "0.5.1+0"

[[deps.RoundingEmulator]]
git-tree-sha1 = "40b9edad2e5287e05bd413a38f61a8ff55b9557b"
uuid = "5eaf0fd0-dfba-4ccb-bf02-d820a40db705"
version = "0.2.1"

[[deps.SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"
version = "0.7.0"

[[deps.ScopedValues]]
deps = ["HashArrayMappedTries", "Logging"]
git-tree-sha1 = "1147f140b4c8ddab224c94efa9569fc23d63ab44"
uuid = "7e506255-f358-4e82-b7e4-beb19740aa63"
version = "1.3.0"

[[deps.Scratch]]
deps = ["Dates"]
git-tree-sha1 = "3bac05bc7e74a75fd9cba4295cde4045d9fe2386"
uuid = "6c6a2e73-6563-6170-7368-637461726353"
version = "1.2.1"

[[deps.Serialization]]
uuid = "9e88b42a-f829-5b0c-bbe9-9e923198166b"
version = "1.11.0"

[[deps.SetRounding]]
git-tree-sha1 = "d7a25e439d07a17b7cdf97eecee504c50fedf5f6"
uuid = "3cc68bcd-71a2-5612-b932-767ffbe40ab0"
version = "0.2.1"

[[deps.Setfield]]
deps = ["ConstructionBase", "Future", "MacroTools", "StaticArraysCore"]
git-tree-sha1 = "e2cc6d8c88613c05e1defb55170bf5ff211fbeac"
uuid = "efcf1570-3423-57d1-acb7-fd33fddbac46"
version = "1.1.1"

[[deps.SharedArrays]]
deps = ["Distributed", "Mmap", "Random", "Serialization"]
uuid = "1a1011a3-84de-559e-8e89-a11a2f7dc383"
version = "1.11.0"

[[deps.Showoff]]
deps = ["Dates", "Grisu"]
git-tree-sha1 = "91eddf657aca81df9ae6ceb20b959ae5653ad1de"
uuid = "992d4aef-0814-514b-bc4d-f2e9a6c4116f"
version = "1.0.3"

[[deps.SignalTemporalLogic]]
deps = ["InteractiveUtils", "Markdown", "PlutoUI", "Zygote"]
git-tree-sha1 = "40e6c51e6d2e7571de6ad1d4dc7e7c94a50f21dc"
uuid = "a79a9ddd-d50e-4d85-a979-5d85760e62a0"
version = "1.0.0"

[[deps.SimpleBufferStream]]
git-tree-sha1 = "f305871d2f381d21527c770d4788c06c097c9bc1"
uuid = "777ac1f9-54b0-4bf8-805c-2214025038e7"
version = "1.2.0"

[[deps.Sockets]]
uuid = "6462fe0b-24de-5631-8697-dd941f90decc"
version = "1.11.0"

[[deps.SortingAlgorithms]]
deps = ["DataStructures"]
git-tree-sha1 = "66e0a8e672a0bdfca2c3f5937efb8538b9ddc085"
uuid = "a2af1166-a08f-5f64-846c-94a0d3cef48c"
version = "1.2.1"

[[deps.SparseArrays]]
deps = ["Libdl", "LinearAlgebra", "Random", "Serialization", "SuiteSparse_jll"]
uuid = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
version = "1.11.0"

[[deps.SparseInverseSubset]]
deps = ["LinearAlgebra", "SparseArrays", "SuiteSparse"]
git-tree-sha1 = "52962839426b75b3021296f7df242e40ecfc0852"
uuid = "dc90abb0-5640-4711-901d-7e5b23a2fada"
version = "0.1.2"

[[deps.SpecialFunctions]]
deps = ["IrrationalConstants", "LogExpFunctions", "OpenLibm_jll", "OpenSpecFun_jll"]
git-tree-sha1 = "64cca0c26b4f31ba18f13f6c12af7c85f478cfde"
uuid = "276daf66-3868-5448-9aa4-cd146d93841b"
version = "2.5.0"
weakdeps = ["ChainRulesCore"]

    [deps.SpecialFunctions.extensions]
    SpecialFunctionsChainRulesCoreExt = "ChainRulesCore"

[[deps.StableRNGs]]
deps = ["Random"]
git-tree-sha1 = "83e6cce8324d49dfaf9ef059227f91ed4441a8e5"
uuid = "860ef19b-820b-49d6-a774-d7a799459cd3"
version = "1.0.2"

[[deps.StanfordAA228V]]
deps = ["AbstractPlutoDingetjes", "BSON", "Base64", "Distributions", "Downloads", "ForwardDiff", "GridInterpolations", "LazySets", "LinearAlgebra", "Markdown", "Optim", "Parameters", "Pkg", "Plots", "Pluto", "PlutoUI", "ProgressLogging", "Random", "SignalTemporalLogic", "Statistics", "TOML"]
git-tree-sha1 = "5376632ae8604432fd7c8ce7308edf70950c1da8"
uuid = "6f6e590e-f8c2-4a21-9268-94576b9fb3b1"
version = "0.1.22"

[[deps.StaticArrays]]
deps = ["LinearAlgebra", "PrecompileTools", "Random", "StaticArraysCore"]
git-tree-sha1 = "47091a0340a675c738b1304b58161f3b0839d454"
uuid = "90137ffa-7385-5640-81b9-e52037218182"
version = "1.9.10"
weakdeps = ["ChainRulesCore", "Statistics"]

    [deps.StaticArrays.extensions]
    StaticArraysChainRulesCoreExt = "ChainRulesCore"
    StaticArraysStatisticsExt = "Statistics"

[[deps.StaticArraysCore]]
git-tree-sha1 = "192954ef1208c7019899fbf8049e717f92959682"
uuid = "1e83bf80-4336-4d27-bf5d-d5a4f845583c"
version = "1.4.3"

[[deps.Statistics]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "ae3bb1eb3bba077cd276bc5cfc337cc65c3075c0"
uuid = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"
version = "1.11.1"
weakdeps = ["SparseArrays"]

    [deps.Statistics.extensions]
    SparseArraysExt = ["SparseArrays"]

[[deps.StatsAPI]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "1ff449ad350c9c4cbc756624d6f8a8c3ef56d3ed"
uuid = "82ae8749-77ed-4fe6-ae5f-f523153014b0"
version = "1.7.0"

[[deps.StatsBase]]
deps = ["AliasTables", "DataAPI", "DataStructures", "LinearAlgebra", "LogExpFunctions", "Missings", "Printf", "Random", "SortingAlgorithms", "SparseArrays", "Statistics", "StatsAPI"]
git-tree-sha1 = "29321314c920c26684834965ec2ce0dacc9cf8e5"
uuid = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"
version = "0.34.4"

[[deps.StatsFuns]]
deps = ["HypergeometricFunctions", "IrrationalConstants", "LogExpFunctions", "Reexport", "Rmath", "SpecialFunctions"]
git-tree-sha1 = "b423576adc27097764a90e163157bcfc9acf0f46"
uuid = "4c63d2b9-4356-54db-8cca-17b64c39e42c"
version = "1.3.2"

    [deps.StatsFuns.extensions]
    StatsFunsChainRulesCoreExt = "ChainRulesCore"
    StatsFunsInverseFunctionsExt = "InverseFunctions"

    [deps.StatsFuns.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    InverseFunctions = "3587e190-3f89-42d0-90ee-14403ec27112"

[[deps.StructArrays]]
deps = ["ConstructionBase", "DataAPI", "Tables"]
git-tree-sha1 = "5a3a31c41e15a1e042d60f2f4942adccba05d3c9"
uuid = "09ab397b-f2b6-538f-b94a-2f83cf4a842a"
version = "0.7.0"
weakdeps = ["Adapt", "GPUArraysCore", "KernelAbstractions", "LinearAlgebra", "SparseArrays", "StaticArrays"]

    [deps.StructArrays.extensions]
    StructArraysAdaptExt = "Adapt"
    StructArraysGPUArraysCoreExt = ["GPUArraysCore", "KernelAbstractions"]
    StructArraysLinearAlgebraExt = "LinearAlgebra"
    StructArraysSparseArraysExt = "SparseArrays"
    StructArraysStaticArraysExt = "StaticArrays"

[[deps.StyledStrings]]
uuid = "f489334b-da3d-4c2e-b8f0-e476e12c162b"
version = "1.11.0"

[[deps.SuiteSparse]]
deps = ["Libdl", "LinearAlgebra", "Serialization", "SparseArrays"]
uuid = "4607b0f0-06f3-5cda-b6b1-a6196a1729e9"

[[deps.SuiteSparse_jll]]
deps = ["Artifacts", "Libdl", "libblastrampoline_jll"]
uuid = "bea87d4a-7f5b-5778-9afe-8cc45184846c"
version = "7.7.0+0"

[[deps.TOML]]
deps = ["Dates"]
uuid = "fa267f1f-6049-4f14-aa54-33bafae1ed76"
version = "1.0.3"

[[deps.TableTraits]]
deps = ["IteratorInterfaceExtensions"]
git-tree-sha1 = "c06b2f539df1c6efa794486abfb6ed2022561a39"
uuid = "3783bdb8-4a98-5b6b-af9a-565f29a5fe9c"
version = "1.0.1"

[[deps.Tables]]
deps = ["DataAPI", "DataValueInterfaces", "IteratorInterfaceExtensions", "OrderedCollections", "TableTraits"]
git-tree-sha1 = "598cd7c1f68d1e205689b1c2fe65a9f85846f297"
uuid = "bd369af6-aec1-5ad0-b16a-f7cc5008161c"
version = "1.12.0"

[[deps.Tar]]
deps = ["ArgTools", "SHA"]
uuid = "a4e569a6-e804-4fa4-b0f3-eef7a1d5b13e"
version = "1.10.0"

[[deps.TensorCore]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "1feb45f88d133a655e001435632f019a9a1bcdb6"
uuid = "62fd8b95-f654-4bbd-a8a5-9c27f68ccd50"
version = "0.1.1"

[[deps.Test]]
deps = ["InteractiveUtils", "Logging", "Random", "Serialization"]
uuid = "8dfed614-e22c-5e08-85e1-65c5234f0b40"
version = "1.11.0"

[[deps.TranscodingStreams]]
git-tree-sha1 = "0c45878dcfdcfa8480052b6ab162cdd138781742"
uuid = "3bb67fe8-82b1-5028-8e26-92a6c54297fa"
version = "0.11.3"

[[deps.Tricks]]
git-tree-sha1 = "7822b97e99a1672bfb1b49b668a6d46d58d8cbcb"
uuid = "410a4b4d-49e4-4fbc-ab6d-cb71b17b3775"
version = "0.1.9"

[[deps.URIs]]
git-tree-sha1 = "67db6cc7b3821e19ebe75791a9dd19c9b1188f2b"
uuid = "5c2747f8-b7ea-4ff2-ba2e-563bfd36b1d4"
version = "1.5.1"

[[deps.UUIDs]]
deps = ["Random", "SHA"]
uuid = "cf7118a7-6976-5b1a-9a39-7adc72f591a4"
version = "1.11.0"

[[deps.UnPack]]
git-tree-sha1 = "387c1f73762231e86e0c9c5443ce3b4a0a9a0c2b"
uuid = "3a884ed6-31ef-47d7-9d2a-63182c4928ed"
version = "1.0.2"

[[deps.Unicode]]
uuid = "4ec0a83e-493e-50e2-b9ac-8f72acf5a8f5"
version = "1.11.0"

[[deps.UnicodeFun]]
deps = ["REPL"]
git-tree-sha1 = "53915e50200959667e78a92a418594b428dffddf"
uuid = "1cfade01-22cf-5700-b092-accc4b62d6e1"
version = "0.4.1"

[[deps.Unitful]]
deps = ["Dates", "LinearAlgebra", "Random"]
git-tree-sha1 = "c0667a8e676c53d390a09dc6870b3d8d6650e2bf"
uuid = "1986cc42-f94f-5a68-af5c-568840ba703d"
version = "1.22.0"

    [deps.Unitful.extensions]
    ConstructionBaseUnitfulExt = "ConstructionBase"
    InverseFunctionsUnitfulExt = "InverseFunctions"

    [deps.Unitful.weakdeps]
    ConstructionBase = "187b0558-2788-49d3-abe0-74a17ed4e7c9"
    InverseFunctions = "3587e190-3f89-42d0-90ee-14403ec27112"

[[deps.UnitfulLatexify]]
deps = ["LaTeXStrings", "Latexify", "Unitful"]
git-tree-sha1 = "975c354fcd5f7e1ddcc1f1a23e6e091d99e99bc8"
uuid = "45397f5d-5981-4c77-b2b3-fc36d6e9b728"
version = "1.6.4"

[[deps.UnsafeAtomics]]
git-tree-sha1 = "b13c4edda90890e5b04ba24e20a310fbe6f249ff"
uuid = "013be700-e6cd-48c3-b4a1-df204f14c38f"
version = "0.3.0"
weakdeps = ["LLVM"]

    [deps.UnsafeAtomics.extensions]
    UnsafeAtomicsLLVM = ["LLVM"]

[[deps.Unzip]]
git-tree-sha1 = "ca0969166a028236229f63514992fc073799bb78"
uuid = "41fe7b60-77ed-43a1-b4f0-825fd5a5650d"
version = "0.2.0"

[[deps.Vulkan_Loader_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Wayland_jll", "Xorg_libX11_jll", "Xorg_libXrandr_jll", "xkbcommon_jll"]
git-tree-sha1 = "2f0486047a07670caad3a81a075d2e518acc5c59"
uuid = "a44049a8-05dd-5a78-86c9-5fde0876e88c"
version = "1.3.243+0"

[[deps.Wayland_jll]]
deps = ["Artifacts", "EpollShim_jll", "Expat_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Pkg", "XML2_jll"]
git-tree-sha1 = "85c7811eddec9e7f22615371c3cc81a504c508ee"
uuid = "a2964d1f-97da-50d4-b82a-358c7fce9d89"
version = "1.21.0+2"

[[deps.Wayland_protocols_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "5db3e9d307d32baba7067b13fc7b5aa6edd4a19a"
uuid = "2381bf8a-dfd0-557d-9999-79630e7b1b91"
version = "1.36.0+0"

[[deps.XML2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libiconv_jll", "Zlib_jll"]
git-tree-sha1 = "a2fccc6559132927d4c5dc183e3e01048c6dcbd6"
uuid = "02c8fc9c-b97f-50b9-bbe4-9be30ff0a78a"
version = "2.13.5+0"

[[deps.XSLT_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libgcrypt_jll", "Libgpg_error_jll", "Libiconv_jll", "XML2_jll", "Zlib_jll"]
git-tree-sha1 = "7d1671acbe47ac88e981868a078bd6b4e27c5191"
uuid = "aed1982a-8fda-507f-9586-7b0439959a61"
version = "1.1.42+0"

[[deps.XZ_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "beef98d5aad604d9e7d60b2ece5181f7888e2fd6"
uuid = "ffd25f8a-64ca-5728-b0f7-c24cf3aae800"
version = "5.6.4+0"

[[deps.Xorg_libICE_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "326b4fea307b0b39892b3e85fa451692eda8d46c"
uuid = "f67eecfb-183a-506d-b269-f58e52b52d7c"
version = "1.1.1+0"

[[deps.Xorg_libSM_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libICE_jll"]
git-tree-sha1 = "3796722887072218eabafb494a13c963209754ce"
uuid = "c834827a-8449-5923-a945-d239c165b7dd"
version = "1.2.4+0"

[[deps.Xorg_libX11_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libxcb_jll", "Xorg_xtrans_jll"]
git-tree-sha1 = "9dafcee1d24c4f024e7edc92603cedba72118283"
uuid = "4f6342f7-b3d2-589e-9d20-edeb45f2b2bc"
version = "1.8.6+3"

[[deps.Xorg_libXau_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "e9216fdcd8514b7072b43653874fd688e4c6c003"
uuid = "0c0b7dd1-d40b-584c-a123-a41640f87eec"
version = "1.0.12+0"

[[deps.Xorg_libXcursor_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libXfixes_jll", "Xorg_libXrender_jll"]
git-tree-sha1 = "807c226eaf3651e7b2c468f687ac788291f9a89b"
uuid = "935fb764-8cf2-53bf-bb30-45bb1f8bf724"
version = "1.2.3+0"

[[deps.Xorg_libXdmcp_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "89799ae67c17caa5b3b5a19b8469eeee474377db"
uuid = "a3789734-cfe1-5b06-b2d0-1dd0d9d62d05"
version = "1.1.5+0"

[[deps.Xorg_libXext_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libX11_jll"]
git-tree-sha1 = "d7155fea91a4123ef59f42c4afb5ab3b4ca95058"
uuid = "1082639a-0dae-5f34-9b06-72781eeb8cb3"
version = "1.3.6+3"

[[deps.Xorg_libXfixes_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libX11_jll"]
git-tree-sha1 = "6fcc21d5aea1a0b7cce6cab3e62246abd1949b86"
uuid = "d091e8ba-531a-589c-9de9-94069b037ed8"
version = "6.0.0+0"

[[deps.Xorg_libXi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libXext_jll", "Xorg_libXfixes_jll"]
git-tree-sha1 = "984b313b049c89739075b8e2a94407076de17449"
uuid = "a51aa0fd-4e3c-5386-b890-e753decda492"
version = "1.8.2+0"

[[deps.Xorg_libXinerama_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libXext_jll"]
git-tree-sha1 = "a1a7eaf6c3b5b05cb903e35e8372049b107ac729"
uuid = "d1454406-59df-5ea1-beac-c340f2130bc3"
version = "1.1.5+0"

[[deps.Xorg_libXrandr_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libXext_jll", "Xorg_libXrender_jll"]
git-tree-sha1 = "b6f664b7b2f6a39689d822a6300b14df4668f0f4"
uuid = "ec84b674-ba8e-5d96-8ba1-2a689ba10484"
version = "1.5.4+0"

[[deps.Xorg_libXrender_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libX11_jll"]
git-tree-sha1 = "a490c6212a0e90d2d55111ac956f7c4fa9c277a6"
uuid = "ea2f1a96-1ddc-540d-b46f-429655e07cfa"
version = "0.9.11+1"

[[deps.Xorg_libpthread_stubs_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "c57201109a9e4c0585b208bb408bc41d205ac4e9"
uuid = "14d82f49-176c-5ed1-bb49-ad3f5cbd8c74"
version = "0.1.2+0"

[[deps.Xorg_libxcb_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "XSLT_jll", "Xorg_libXau_jll", "Xorg_libXdmcp_jll", "Xorg_libpthread_stubs_jll"]
git-tree-sha1 = "1a74296303b6524a0472a8cb12d3d87a78eb3612"
uuid = "c7cfdc94-dc32-55de-ac96-5a1b8d977c5b"
version = "1.17.0+3"

[[deps.Xorg_libxkbfile_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libX11_jll"]
git-tree-sha1 = "dbc53e4cf7701c6c7047c51e17d6e64df55dca94"
uuid = "cc61e674-0454-545c-8b26-ed2c68acab7a"
version = "1.1.2+1"

[[deps.Xorg_xcb_util_cursor_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_xcb_util_image_jll", "Xorg_xcb_util_jll", "Xorg_xcb_util_renderutil_jll"]
git-tree-sha1 = "04341cb870f29dcd5e39055f895c39d016e18ccd"
uuid = "e920d4aa-a673-5f3a-b3d7-f755a4d47c43"
version = "0.1.4+0"

[[deps.Xorg_xcb_util_image_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "0fab0a40349ba1cba2c1da699243396ff8e94b97"
uuid = "12413925-8142-5f55-bb0e-6d7ca50bb09b"
version = "0.4.0+1"

[[deps.Xorg_xcb_util_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libxcb_jll"]
git-tree-sha1 = "e7fd7b2881fa2eaa72717420894d3938177862d1"
uuid = "2def613f-5ad1-5310-b15b-b15d46f528f5"
version = "0.4.0+1"

[[deps.Xorg_xcb_util_keysyms_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "d1151e2c45a544f32441a567d1690e701ec89b00"
uuid = "975044d2-76e6-5fbe-bf08-97ce7c6574c7"
version = "0.4.0+1"

[[deps.Xorg_xcb_util_renderutil_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "dfd7a8f38d4613b6a575253b3174dd991ca6183e"
uuid = "0d47668e-0667-5a69-a72c-f761630bfb7e"
version = "0.3.9+1"

[[deps.Xorg_xcb_util_wm_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "e78d10aab01a4a154142c5006ed44fd9e8e31b67"
uuid = "c22f9ab0-d5fe-5066-847c-f4bb1cd4e361"
version = "0.4.1+1"

[[deps.Xorg_xkbcomp_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libxkbfile_jll"]
git-tree-sha1 = "ab2221d309eda71020cdda67a973aa582aa85d69"
uuid = "35661453-b289-5fab-8a00-3d9160c6a3a4"
version = "1.4.6+1"

[[deps.Xorg_xkeyboard_config_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_xkbcomp_jll"]
git-tree-sha1 = "691634e5453ad362044e2ad653e79f3ee3bb98c3"
uuid = "33bec58e-1273-512f-9401-5d533626f822"
version = "2.39.0+0"

[[deps.Xorg_xtrans_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "6dba04dbfb72ae3ebe5418ba33d087ba8aa8cb00"
uuid = "c5fb5394-a638-5e4d-96e5-b29de1b5cf10"
version = "1.5.1+0"

[[deps.Zlib_jll]]
deps = ["Libdl"]
uuid = "83775a58-1f1d-513f-b197-d71354ab007a"
version = "1.2.13+1"

[[deps.Zstd_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "622cf78670d067c738667aaa96c553430b65e269"
uuid = "3161d3a3-bdf6-5164-811a-617609db77b4"
version = "1.5.7+0"

[[deps.Zygote]]
deps = ["AbstractFFTs", "ChainRules", "ChainRulesCore", "DiffRules", "Distributed", "FillArrays", "ForwardDiff", "GPUArrays", "GPUArraysCore", "IRTools", "InteractiveUtils", "LinearAlgebra", "LogExpFunctions", "MacroTools", "NaNMath", "PrecompileTools", "Random", "Requires", "SparseArrays", "SpecialFunctions", "Statistics", "ZygoteRules"]
git-tree-sha1 = "0b3c944f5d2d8b466c5d20a84c229c17c528f49e"
uuid = "e88e6eb3-aa80-5325-afca-941959d7151f"
version = "0.6.75"

    [deps.Zygote.extensions]
    ZygoteColorsExt = "Colors"
    ZygoteDistancesExt = "Distances"
    ZygoteTrackerExt = "Tracker"

    [deps.Zygote.weakdeps]
    Colors = "5ae59095-9a9b-59fe-a467-6f913c188581"
    Distances = "b4f34e82-e78d-54a5-968a-f98e89d6e8f7"
    Tracker = "9f7883ad-71c0-57eb-9f7f-b5c9e6d3789c"

[[deps.ZygoteRules]]
deps = ["ChainRulesCore", "MacroTools"]
git-tree-sha1 = "27798139afc0a2afa7b1824c206d5e87ea587a00"
uuid = "700de1a5-db45-46bc-99cf-38207098b444"
version = "0.2.5"

[[deps.eudev_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "gperf_jll"]
git-tree-sha1 = "431b678a28ebb559d224c0b6b6d01afce87c51ba"
uuid = "35ca27e7-8b34-5b7f-bca9-bdc33f59eb06"
version = "3.2.9+0"

[[deps.fzf_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "6e50f145003024df4f5cb96c7fce79466741d601"
uuid = "214eeab7-80f7-51ab-84ad-2988db7cef09"
version = "0.56.3+0"

[[deps.gperf_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "0ba42241cb6809f1a278d0bcb976e0483c3f1f2d"
uuid = "1a1c6b14-54f6-533d-8383-74cd7377aa70"
version = "3.1.1+1"

[[deps.libaom_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "522c1df09d05a71785765d19c9524661234738e9"
uuid = "a4ae2306-e953-59d6-aa16-d00cac43593b"
version = "3.11.0+0"

[[deps.libass_jll]]
deps = ["Artifacts", "Bzip2_jll", "FreeType2_jll", "FriBidi_jll", "HarfBuzz_jll", "JLLWrappers", "Libdl", "Zlib_jll"]
git-tree-sha1 = "e17c115d55c5fbb7e52ebedb427a0dca79d4484e"
uuid = "0ac62f75-1d6f-5e53-bd7c-93b484bb37c0"
version = "0.15.2+0"

[[deps.libblastrampoline_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850b90-86db-534c-a0d3-1478176c7d93"
version = "5.11.0+0"

[[deps.libdecor_jll]]
deps = ["Artifacts", "Dbus_jll", "JLLWrappers", "Libdl", "Libglvnd_jll", "Pango_jll", "Wayland_jll", "xkbcommon_jll"]
git-tree-sha1 = "9bf7903af251d2050b467f76bdbe57ce541f7f4f"
uuid = "1183f4f0-6f2a-5f1a-908b-139f9cdfea6f"
version = "0.2.2+0"

[[deps.libevdev_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "141fe65dc3efabb0b1d5ba74e91f6ad26f84cc22"
uuid = "2db6ffa8-e38f-5e21-84af-90c45d0032cc"
version = "1.11.0+0"

[[deps.libfdk_aac_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "8a22cf860a7d27e4f3498a0fe0811a7957badb38"
uuid = "f638f0a6-7fb0-5443-88ba-1cc74229b280"
version = "2.0.3+0"

[[deps.libinput_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "eudev_jll", "libevdev_jll", "mtdev_jll"]
git-tree-sha1 = "ad50e5b90f222cfe78aa3d5183a20a12de1322ce"
uuid = "36db933b-70db-51c0-b978-0f229ee0e533"
version = "1.18.0+0"

[[deps.libpng_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Zlib_jll"]
git-tree-sha1 = "d7b5bbf1efbafb5eca466700949625e07533aff2"
uuid = "b53b4c65-9356-5827-b1ea-8c7a1a84506f"
version = "1.6.45+1"

[[deps.libvorbis_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Ogg_jll", "Pkg"]
git-tree-sha1 = "490376214c4721cdaca654041f635213c6165cb3"
uuid = "f27f6e37-5d2b-51aa-960f-b287f2bc3b7a"
version = "1.3.7+2"

[[deps.mtdev_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "814e154bdb7be91d78b6802843f76b6ece642f11"
uuid = "009596ad-96f7-51b1-9f1b-5ce2d5e8a71e"
version = "1.1.6+0"

[[deps.nghttp2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850ede-7688-5339-a07c-302acd2aaf8d"
version = "1.59.0+0"

[[deps.p7zip_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "3f19e933-33d8-53b3-aaab-bd5110c3b7a0"
version = "17.4.0+2"

[[deps.x264_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "4fea590b89e6ec504593146bf8b988b2c00922b2"
uuid = "1270edf5-f2f9-52d2-97e9-ab00b5d0237a"
version = "2021.5.5+0"

[[deps.x265_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "ee567a171cce03570d77ad3a43e90218e38937a9"
uuid = "dfaa095f-4041-5dcd-9319-2fabd8486b76"
version = "3.5.0+0"

[[deps.xkbcommon_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Wayland_jll", "Wayland_protocols_jll", "Xorg_libxcb_jll", "Xorg_xkeyboard_config_jll"]
git-tree-sha1 = "63406453ed9b33a0df95d570816d5366c92b7809"
uuid = "d8fb68d0-12a3-5cfd-a85a-d49703b185fd"
version = "1.4.1+2"
"""

# ╔═╡ Cell order:
# ╟─8c18963c-e916-4dd0-8c1a-9ded8434d1a2
# ╠═b3cd799e-17fe-4c80-80e1-107c6e64cc31
# ╟─f1a3a270-073b-11eb-1741-37897aa84974
# ╟─6efb1630-074c-11eb-3186-b3ea3cc6d33b
# ╟─44fb994a-100c-4009-a064-99cf5b667e64
# ╟─776449d6-2c4c-450f-a163-21b2290e4a12
# ╠═8005fa69-78ea-40a8-9cda-8ea33470f899
# ╠═433a69f0-074d-11eb-0698-f9113d9939c3
# ╠═4a4f2872-074d-11eb-0ec8-13783b11ffd7
# ╠═5541c6ea-20cb-421f-9be9-009aebd80e6c
# ╟─baf23507-63a7-472e-933c-e100b8cfae51
# ╟─3bf21370-bfda-412b-adbf-dafaf42a471d
# ╟─991297c0-0776-11eb-082c-e57372352faa
# ╠═0f1aac87-102d-4fec-b012-e364a7a23b0f
# ╟─50caeb00-073c-11eb-36ac-5b8839bb70ab
# ╠═5996f670-073c-11eb-3a63-67246e676f4e
# ╠═5fe5f3f0-073c-11eb-33ae-2343a63d952d
# ╠═62961fce-073c-11eb-315e-03a405d69157
# ╠═8ca1d3a0-073c-11eb-2d51-7766203bdf92
# ╠═90e6d04e-073c-11eb-0a64-a5596e6b6079
# ╠═9516be60-073c-11eb-277b-b59f20b2feba
# ╠═9895d300-073c-11eb-1fe4-d3337747efcd
# ╠═9be413a0-073c-11eb-3c73-df78ea75bcd1
# ╠═a10c4462-073c-11eb-31f8-6f675614356d
# ╠═a4de820e-073c-11eb-374e-efd08bbc884c
# ╠═ae65ee42-073c-11eb-1dbd-eb918f086ab7
# ╟─b427431e-e220-416f-974f-791c43bf0c18
# ╠═238efdef-6820-48dd-a24f-9b7c8f108f5d
# ╠═92fb38fc-c10d-4a66-9bfe-dfa73f3a7996
# ╠═9141fbe7-5250-4aff-8754-49a5c9e1884e
# ╟─b4ad4aa0-073c-11eb-238e-a913116c0944
# ╠═bf0155a2-073c-11eb-1ff2-e9d78baf273a
# ╠═c22997b0-073c-11eb-31c8-952711ee4422
# ╠═c393d612-073c-11eb-071d-93ca1d801f4d
# ╠═c59412e0-073c-11eb-0b89-7d14cda40917
# ╠═cd146470-073c-11eb-131c-250b433dbe74
# ╟─d5dc2a20-073c-11eb-1667-038dd7a06c63
# ╠═e05151b0-073c-11eb-2cd2-75f9f350b266
# ╠═69601831-6265-4341-beed-4c01a54ed705
# ╟─e90af052-709d-4587-9982-f224afd49af2
# ╠═ddaf4dc2-d5a6-458e-a7f4-280bf64663f3
# ╠═4d0b3fd3-a15d-48fc-9219-cd6e02bb5b03
# ╟─e3846930-073c-11eb-384a-b1b2e27d16cc
# ╠═ef0693f0-073c-11eb-14da-0bec0f5bfe2e
# ╠═f131d862-073c-11eb-0584-1947b568926c
# ╟─f5e1a4d0-073c-11eb-0002-d94d6a932b0b
# ╠═004e40e0-073d-11eb-1475-5dea9854afd4
# ╠═781dcaf0-073d-11eb-2498-0f3bdb572f88
# ╟─80309880-073d-11eb-10d0-53a70045661f
# ╠═7a9cd4ae-073d-11eb-1c20-f1a0bc572b33
# ╠═db37f460-0744-11eb-24b0-cb3767f0cf44
# ╟─8ea1a5d0-073d-11eb-029b-01b0401da847
# ╠═993a33e0-073d-11eb-2ded-4fc896fd19d7
# ╠═e95927ed-cfc5-428d-87e1-addc0983d47b
# ╠═4982024e-9e0e-4968-a521-18256e5ead5a
# ╟─f99743b0-0745-11eb-049e-71c7b72884d1
# ╟─fbaad713-216b-4557-8132-3a1e95ed0a27
# ╠═4a2f40a4-029f-43bb-9f55-3bf001861d0c
# ╟─d3d5cde4-8d81-45b1-8554-4ee2e009074c
# ╠═efa419b0-40ce-41ac-a660-5bd1743b0e6c
# ╠═bc755cd7-40ca-4ad6-9f88-8ff657e4e397
# ╠═a0ffcbd0-073d-11eb-3c0a-bfc967428073
# ╠═4968e820-0742-11eb-1b1b-034751f95fb9
# ╟─c5c2e4a7-6a4f-4d06-9cc5-0011dafbffe3
# ╠═d50862fe-4e9d-4dd8-9278-1f935ace223b
# ╠═b1ed3fa0-b885-48be-9332-653623d4b606
# ╟─62c9ce6e-0746-11eb-0911-afc23d8b351c
# ╠═64f36530-0746-11eb-3c66-59091a9c7d7d
# ╟─66335f40-0746-11eb-2f7e-ffe20c76f21f
# ╠═71dc24d0-0746-11eb-1eac-adcbb393c38b
# ╟─765e5190-0746-11eb-318e-8954e5e8fa3e
# ╠═81956990-0746-11eb-2ca4-63ba1d192b97
# ╟─85101160-0746-11eb-1501-3101c2006157
# ╠═97a54cf0-0746-11eb-391d-d70312796ded
# ╠═9a808070-0746-11eb-05cf-81547eab646d
# ╠═9d560e9e-0746-11eb-1e55-55e827e7423d
# ╠═9f1264a0-0746-11eb-2554-1f50ba874f57
# ╠═a2283020-0746-11eb-341a-fb4e280be5d6
# ╠═a9ea93c0-0746-11eb-0956-95d12cb066ac
# ╠═bc8dd90e-0746-11eb-3c30-1b27e08fd17d
# ╠═ad279650-0746-11eb-2090-81a679e5f3be
# ╠═b1266240-0746-11eb-262c-893974d49c9f
# ╠═b5a1b130-0746-11eb-2038-d353aad7e355
# ╠═c4f7ee60-0746-11eb-324b-854c3b2e383e
# ╠═c7484700-0746-11eb-3327-f321a4423d2a
# ╠═cb5d3300-0746-11eb-35d3-33280e451394
# ╠═cd941030-0746-11eb-34d7-216aa0f8f33d
# ╠═d6663620-0746-11eb-01dc-27b5e3b11ab8
# ╠═da28e370-0746-11eb-1ea1-91664661a74d
# ╠═df66e620-0746-11eb-37b9-d3f3ab3dd12f
# ╠═e3e171c0-0746-11eb-0e1d-c3fc24347d47
# ╟─e99fa0f0-0746-11eb-1f5c-7da5d6765131
# ╠═e5afc920-0746-11eb-0558-79599697bec6
# ╠═e8dc9f10-0746-11eb-2c56-c9383000043c
# ╠═1b239000-0747-11eb-0a63-d9e58c6dfda3
# ╠═24b05360-0747-11eb-0783-ab42074819c4
# ╟─410c4e10-0747-11eb-0acf-116ff6073047
# ╠═4842ec70-0747-11eb-37b0-21da3d5049ff
# ╠═7351b860-0747-11eb-16c5-833309f7fbcb
# ╠═750c87c0-0747-11eb-1aeb-d32e03b686f5
# ╟─7f9e37fe-0747-11eb-295e-6d55a31d8395
# ╠═77cd074e-0747-11eb-0306-05ff0e6ada53
# ╠═88232852-0747-11eb-289d-1742e687b041
# ╠═3589dfc0-0748-11eb-1b7f-672e2f6dcf53
# ╠═48a9a810-0748-11eb-0b15-e9085ebd7b52
# ╠═49c63ba0-0748-11eb-158d-57dcd4ad537d
# ╠═4a742ee0-0748-11eb-364c-f7eb2b89d88a
# ╠═64a33770-0748-11eb-1221-b994ffb70091
# ╠═668d060e-0748-11eb-0f34-f307c94e755d
# ╠═6722dd70-0748-11eb-31b7-d38b56f4cc0f
# ╟─7226fe92-0748-11eb-215e-49075766b2da
# ╠═5bfd2a90-0748-11eb-3b5c-8f191bd23f1c
# ╠═7f8e8b20-0748-11eb-39da-435f6c49934a
# ╟─84d7d870-0748-11eb-2a11-5797476719b5
# ╠═8dc1a510-0748-11eb-1e2d-ab6fc445d549
# ╠═90afc450-0748-11eb-170e-9fd33246ec06
# ╠═a28f9290-0748-11eb-0e3a-539a124905c0
# ╠═a6f27780-0748-11eb-2ca3-69c3f2923f7e
# ╠═a8c3b510-0748-11eb-39b2-7389d0ee67e4
# ╠═ac35d160-0748-11eb-00c4-e799f1c83746
# ╠═b137dc80-0748-11eb-3e3b-d9eb6ade524c
# ╠═b47d6a92-0748-11eb-1a5c-fb5faaf20c14
# ╟─bb233aa0-0748-11eb-3488-8d316224bdf8
# ╠═d2ec8dd0-0748-11eb-298a-5d94d5da2477
# ╟─db2d2220-0748-11eb-0889-a30e49f2d784
# ╠═e891a170-0748-11eb-2bbd-e15baa27423c
# ╟─e129e5c7-74f5-4e5e-8c99-cc99217395c4
# ╠═6f4df9f0-9296-4b8f-a81b-69677cebd6e0
# ╠═1470bb4b-f649-4fc8-8e73-0b04e439f9a2
# ╟─da7ac1f5-ea05-4d88-af31-45f8d11356d2
# ╠═f2b5413f-9b8e-4926-a6ee-7b3142f53dcd
# ╟─38ebf397-acfa-4f76-86ea-f6e96cd8fb9a
# ╟─31a01337-d09d-421f-b217-ae85d38562ee
# ╟─2f09cab2-0749-11eb-3533-79dae3c99545
# ╠═204a7650-0749-11eb-3bcf-2d2846eb951b
# ╠═50260c90-0749-11eb-0af7-8fa4ca5e3890
# ╟─245f5636-a5a7-477b-bf69-b46ad3e07a39
# ╠═69c88c90-0749-11eb-1c23-a5f0042bf2de
# ╟─1c3b59e6-a94b-4d58-8485-5d3cd0285312
# ╟─fef5c85a-8536-42aa-b169-4b5ccab6e6bd
# ╠═870a7a70-0749-11eb-2ab1-e9279dd4642a
# ╟─9ef92ff2-0749-11eb-119e-35edd2a409c4
# ╟─c3ac5430-0749-11eb-13f0-cde5ec9409cb
# ╠═a12781a0-0749-11eb-0019-3154576cfbc5
# ╟─514268b6-e65c-4ba7-b009-c0398e31c890
# ╠═ba349f70-0749-11eb-3a4a-294a8c484463
# ╟─d6c45450-0749-11eb-3a9c-41cbc41c8d08
# ╠═dedb4090-0749-11eb-38f4-7dffd22ae8c5
# ╠═e9fd49f0-0749-11eb-3cf6-b78a95067ee3
# ╠═f1339a30-0749-11eb-0fcb-21c6e9917eb9
# ╠═f6631df0-0749-11eb-111d-270176d2bd76
# ╟─f81b0720-0749-11eb-2217-9dd2714570b3
# ╠═06186b12-074a-11eb-2ff8-d7ecf3b88f3b
# ╠═1ae6cdc0-074a-11eb-2415-710522e2ff61
# ╠═1f2031b0-074a-11eb-2f30-43fa1403837e
# ╠═2547f820-074a-11eb-2aa5-ffe306f6b1e2
# ╟─318a3e2c-d71e-4527-9a97-60bf6fb0a7de
# ╟─28c44da0-074a-11eb-07cf-21435e264c3b
# ╠═39c7ecb0-074a-11eb-0f6a-59d57a454722
# ╟─38f57f03-db11-43b1-becb-d3463143ce8a
# ╠═40e16622-074a-11eb-1f0d-579043dff6df
# ╟─03c4df16-b919-487d-89b2-acff111482e0
# ╟─71088c86-7253-4b0b-b5eb-86533ca98db1
# ╟─de4bc9a3-fcf4-434e-add3-599b4676b267
# ╟─4602b910-074a-11eb-3b77-dd28974218f6
# ╠═4e6b95e0-074a-11eb-324c-09c41dd1fb64
# ╟─55249fd0-074a-11eb-1df3-2bfaa49155ae
# ╠═60e50c10-074a-11eb-0249-339b5ee9bcf2
# ╠═6cc00530-074a-11eb-08da-5f69fb9e6c08
# ╠═72681450-074a-11eb-2182-9ba3de8ae5c7
# ╠═89e51c40-074a-11eb-199f-79a721854c1f
# ╟─f1055123-309a-4ee1-aa48-204eb0308b8f
# ╠═fd41492e-9a13-47d1-8f9f-b55edf4a93f3
# ╠═4e2c0e4a-64be-4bf6-972a-3f4b1d236fb2
# ╟─e62935c6-8d25-4c2f-a04f-5f7160347d99
# ╟─ca41086a-8f5e-4617-a4cf-5cf5818ba8bf
# ╠═84b18b58-da13-408b-8e00-ead7265a9bd4
# ╠═6f1aed26-98a4-4488-b25c-8a1aab4c0050
# ╟─16693bfd-dc5c-4526-9fff-e8d488638420
# ╠═d470ec22-bdff-44bc-ba9c-e829e4d57101
# ╠═5304e5f0-d76b-4b99-bf9d-f865bb41b37e
# ╠═5550f8c3-23c1-4822-82cf-1eaae478f9dd
# ╟─90db2f30-074a-11eb-2990-112df2b43ff3
# ╠═28c3d9f6-8666-44fa-b536-7c05149630eb
# ╠═755f8671-9076-4f4f-bbbf-1bd214a2d0a9
# ╠═6ed95d98-7d68-41a3-9b01-1fee030138cc
# ╟─f6771f21-2c0a-40f6-b047-5850e3b2d26e
# ╟─bda58dd0-074a-11eb-37e9-a918c670d380
# ╠═b45a403e-074a-11eb-1144-fd4d939b8bc8
# ╟─1e3f98f4-8333-47c0-9936-76400633affd
# ╠═ceb62530-074a-11eb-2d0f-7383bc2bb7ea
# ╠═d37ea9c0-074a-11eb-075b-ef2a3aa472d0
# ╟─f447e9a0-074a-11eb-1a5b-738a852d47a0
# ╠═2754b1f0-0914-11eb-1052-1159804bcc1c
# ╠═0bb9a6f0-074b-11eb-0cd5-55f6817db5fd
# ╠═34216ec0-074b-11eb-0ec5-b933ea8ecf34
# ╟─9dde0a7b-bbca-4970-a1ea-db47bbdfa102
# ╟─573a1ce0-074b-11eb-2c5d-8ddb9d0c07ed
# ╠═3827f232-0914-11eb-3365-35e127a537ce
# ╠═c26ed0e7-1883-4f82-a13c-005def6e78cd
# ╠═b2b1a3e2-074b-11eb-1a3d-3fb4f9c09ba9
# ╟─3a2f2748-744b-4a96-beaa-3036a7df7765
# ╠═5d31d2a2-074b-11eb-169a-a7423f75a9e6
# ╠═b7cc6720-074b-11eb-31e0-13dea28d37ec
# ╟─e534d70a-af4d-4d4b-b844-a5e055af93f2
# ╠═9ae8c099-0131-4022-bbc5-c10e78ea3e8d
# ╠═da8fe338-68db-4981-af99-07baa5e919bc
# ╠═f8f79331-36ef-4705-93a2-1a5d0da3413e
# ╠═43902359-b6eb-4eb0-aa20-dc3cf913d80e
# ╟─3ab40be6-9b96-42f8-9a06-0fabd73c8a07
# ╟─203a9c7c-ab82-4ae0-8677-c6ac7b03f73b
# ╠═1488d038-e6ea-4b6d-951f-c424b3983d11
# ╟─9bfecb56-d6d4-4f70-9fb4-a8c0d116136d
# ╟─7cb2d32a-12ed-45ae-aeef-da12ed65eaa6
# ╠═457a76fb-0d7a-48ee-9405-3685c8281381
# ╠═f6054e75-fcfe-4e40-a33c-2c3feefbd2ff
# ╠═5512d8cd-d9e8-48eb-b720-f6399b226dd6
# ╠═8fa3ecd9-a81a-4789-9ce7-8f2602807643
# ╟─cd91a50f-bca9-40ca-a2f4-2e3186d38aee
# ╟─49dcf1dd-536b-446b-9673-5f92dcd7a96c
# ╠═b1c2f69a-3e12-40f5-a568-c84537809d64
# ╟─f776a943-79e1-4b69-9501-c188d2435520
# ╠═7fb98a12-b67b-4aad-8489-d062be92c946
# ╠═44344661-eead-430a-8cd0-99001412fbdc
# ╠═659e37a5-8cdc-4a75-b947-7e55f7b2252b
# ╠═d804e977-b6b3-4d82-b840-bbda518b5183
# ╠═42361e99-0cf5-4c2e-b4b9-313556b29941
# ╠═4b053619-2b4c-421d-bc9b-cd16e6203acc
# ╠═55d0aa5c-b864-4360-acea-5995647b8965
# ╟─29415c3e-1353-4498-b940-07175b3e15d9
# ╠═58dece25-d00a-4bf5-bbe1-2eb874e842d2
# ╠═5d1786e1-94b9-46d8-8f9c-064201eb88a7
# ╟─cfb8f437-e897-4688-90b7-c89aa3cf76d9
# ╠═3adc20d5-684a-42c4-a4f4-51a641b41cf2
# ╠═16735a49-1784-41ce-9653-856f0954ae9a
# ╟─c864791d-81fc-4666-87cd-542bc6beb06f
# ╠═f834844a-33b7-4483-a73f-78ab43956d25
# ╠═1756e1fc-df99-4696-9007-485c807a4b91
# ╠═7c1cc3c2-1447-4d5d-a100-4b232b00ad32
# ╠═61c633dd-7572-4fea-918d-0f3ec5674822
# ╟─c0ecc29f-7760-457b-b290-e8fcb67c7875
# ╟─a1c0fbdd-3ced-4644-8f50-f3dba1e8ec10
# ╟─d1d028eb-9872-4f04-8416-9fa0da0d01ff
# ╠═4dc6fdc5-3cf1-475d-8f1d-41640dd5dcb4
# ╟─988b8e06-f5c0-4882-a50f-d96e5d218310
# ╠═0e51c879-fc04-4af0-8325-01a55c4e9082
# ╟─c68b68e9-078d-4300-beba-49d37206902b
# ╠═65cc6f91-d273-4134-9bcb-2485b95dd1d9
# ╟─e026d4f7-6ed8-43fa-b146-7cc3658ae372
# ╟─75bd1e1a-30f4-479b-829d-3a802a12a0f5
# ╠═80d610cf-f2d8-4c6f-993c-a83c180a7d43
# ╟─c3a7e709-4fdf-429d-8217-ec65abf9b350
# ╠═5cd78123-0312-472e-bfa3-9e166e54b89f
# ╟─0d994f6e-f874-4d7b-8335-a2e09184aadd
# ╟─fa03914b-1603-455c-a437-d17eda6fcee2
# ╟─c9a1900c-3a99-47e7-ad2e-9eb98504a3c9
# ╠═7d42e44e-7a4f-4325-aec8-ecda3bd2a751
# ╟─bee6191c-d553-44e6-952b-ed1744d735a0
# ╟─89d08f17-9fc0-4665-a500-ac5967deedc5
# ╠═1a4a6777-c031-40bb-a01a-0e59226f22b3
# ╠═2d1b88f7-91dd-490f-99c2-22731313fab7
# ╟─2e29d581-7281-41c4-a079-79f951a6f0ec
# ╟─8d6ddcfa-6e72-4a5f-a8d7-bbeacd6b470b
# ╠═c8748922-8ca5-486e-93ae-8d730e4a7c69
# ╠═400a841e-e742-4b6f-9277-1648b58079e9
# ╟─0cebdf5d-e2e3-4131-a623-79581b22898f
# ╠═add31696-8a6d-4808-8ac8-91429768cab0
# ╠═34629b2d-defe-4bd1-8d29-ab59c4f7299b
# ╟─a7aa0880-d88a-42d8-9cf9-892210486aef
# ╟─89b956a9-05ec-4120-9469-468270315881
# ╠═29a7d282-7235-447f-bd6d-3f5b6f4df57c
# ╠═edeaff05-3f37-41f2-b10e-7530c0bbb301
# ╠═0f13d11c-b580-47cd-8356-93fead9e9f2f
# ╠═d84184b9-b9bb-4552-9471-f22931242aa5
# ╟─d4243a61-d0a0-4f07-8a02-abcbda51a5d8
# ╠═448a2014-33ab-4043-aa5c-aa1a6a763299
# ╠═e56132c3-cd06-4f3c-9e18-31b30b39163c
# ╟─f6c78515-7efe-429f-94df-9413f6974410
# ╟─49e1b7c3-7ae0-480f-acc6-0ec08d91564e
# ╠═fc607c2c-e153-461b-9e00-e7e478e3266f
# ╠═92f6133c-1e19-4b96-b43b-58bc67fbcfe8
# ╠═822610e1-d90b-46d0-a78b-7661cb492299
# ╠═d5318314-8569-4652-a2b7-20eca1c7c6cd
# ╠═5808f5d3-8bbd-4df6-8502-0e40335922e4
# ╟─edc61fc4-5123-439e-ab0d-29fc8cee5cf2
# ╟─cd775016-1af0-494b-b8d6-abc98afd8cae
# ╠═0cc2cd07-6de7-4c51-bbf3-a2a84911d0cc
# ╠═2e5fc949-bab1-42ab-9543-0fb7342ec6c6
# ╟─cb05d040-c5a8-44e9-8e98-a462452cdd09
# ╠═47e9e011-1927-470b-a1e3-6ebd866f4b3b
# ╟─73b267d5-1eb6-48da-ae75-65a21ae64e40
# ╠═1e7f5155-9bf4-41ca-8d0d-337828a7466c
# ╟─150db79c-d2ac-4227-949b-8f448aad7703
# ╟─6ffac9f1-7053-4b9a-812c-1f483685e7f0
# ╠═dab00858-fa79-46fb-8472-262b18f9c2da
# ╟─b6188b39-b3a4-4a96-8f82-b624abe99029
# ╟─f8a0bc8f-5f31-40da-8964-29113cae9bae
# ╠═30a336af-8d9f-448e-b763-d4dbec4f68a5
# ╟─ba74f9d2-cf25-4f8a-99cb-7ba3ad7fa594
# ╟─2ee67812-ecb2-4d53-a890-ea398e88e36c
# ╠═d8a738c6-6264-48d1-8aa4-6147ee2b7898
# ╟─da4027dc-67e7-4297-9804-723c95296704
# ╟─2f2a0f8b-53b5-4283-b2b6-8735e4071b68
# ╠═959910e8-02ce-4b31-bd0d-5aa5ec3cc03f
# ╟─d7a032b6-5023-436a-8407-5e6e7684a241
# ╟─5dd79519-5522-40b2-965c-603e28d1126d
# ╠═f78c40e9-8299-4397-9052-9390805b0193
# ╠═42226951-148b-4aa9-979c-8f8bef22ce6f
# ╟─cfb583d9-af6e-4129-bae7-0aaa513ded3d
# ╠═f9b050ae-2b92-4274-8d35-15bb29ad0162
# ╟─21bc516e-8000-4e5c-93c6-e1b9a3ec782e
# ╠═b23a65ba-5ebc-48bc-b013-539af1d28b65
# ╟─7fc71f67-0b41-4dda-abf1-ba6db3d027b4
# ╠═6d34e3cd-8ca0-44fe-852d-3d09922802dc
# ╠═b1b0037a-f058-4bac-9d8f-038260c0eff1
# ╠═f07431f4-984f-4882-8962-3cbbeaa1007a
# ╟─2d5920cb-c879-44e8-855d-8bd574c0f4bb
# ╠═215be802-1998-4887-86a6-3bb8e6e5128c
# ╟─4bbe0e01-b27d-4247-b294-42892db75702
# ╠═b0e214ce-7283-4f97-a956-cd218473762f
# ╠═15403d6b-6e2b-488e-806a-9e9d7ebbfb94
# ╠═c0ff7327-ea30-4d74-9f17-41309b1abf40
# ╠═134c7eb8-e52b-4dc3-bdcd-fb8bb0a6ecfd
# ╟─87682e53-7286-4a17-9ee4-2f725cd508a0
# ╠═cb3646a1-4d12-4087-8fad-18e44c357623
# ╟─3637fed7-5acc-49ad-8ec0-b7b48ea91d55
# ╟─f191673f-236f-44a0-a7d6-e47264c4b6c1
# ╟─ed3b1a88-930f-4b3f-aadf-a29da0c7d953
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
