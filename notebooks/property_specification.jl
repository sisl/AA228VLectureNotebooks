### A Pluto.jl notebook ###
# v0.20.3

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

# ╔═╡ 480491fb-9f19-49ee-8f0f-0bd8c1c352d8
begin
	using Pkg
	Pkg.activate()
	using StanfordAA228V
	using PlutoUI
	using Distributions
	using LinearAlgebra
	using Plots
	using Random
	using LazySets
	using PlutoPapers
	using SignalTemporalLogic

	default(fontfamily="Computer Modern", framestyle=:box) # LaTeX-style plotting
	theblue = RGB(128 / 255, 185 / 255, 255 / 255)
	thepurple = RGB(195 / 255, 184 / 255, 255 / 255)
	nothing
end

# ╔═╡ 9f76c16b-6564-4614-b362-1b4a4804c89e
begin
	presentation = PlutoPaper(
		documentclass=Tufte(),
		title="Algorithms for Validation: Property Specification",
		authors=[
			# Author(name="Lecture Introduction")
			# Author(name="Mykel Kochenderfer")
			Author(name="Sydney Katz")
			# Author(name="Anthony Corso")
			# Author(name="Robert Moss")
		]
	)
	
	applyclass(presentation.documentclass)
end

# ╔═╡ f7456e13-d5e8-4064-a138-e3c77ab1e106
title(presentation)

# ╔═╡ 1cfcbf7c-fa09-4755-bc3a-e3666a7fb1cb
@section "Risk Metrics"

# ╔═╡ 5d81cee1-9eb8-4592-9aae-ecb79b8d106f
@section "Preference Elicitation"

# ╔═╡ a4848c23-17c3-42f0-87c9-13e9cdeb47bc
function get_halfspace(query, pref)
	if pref == "a"
		b = query[2]
		a = query[1]
	else
		b = query[1]
		a = query[2]
	end
	return HalfSpace([a[1]-a[3]-b[1]+b[3], a[2]-a[3]-b[2]+b[3]], b[3]-a[3])
end;

# ╔═╡ 748f98c8-ed69-4021-925e-a59641b3115e
md"""
In this case, we preselected the queries. You can imagine that we would probably do better if we selected the queries as we went to maximize the amount of information they provide us! This is the topic of lots of research on preference-based learning. (beyond the scope of this class)
"""

# ╔═╡ e45364dd-ecaf-4b49-b98e-13f6670be244
md"""
We also assumed that the expert was perfectly rational. It is common to relax this assumption and instead compute a distribution over the weights. (beyond the scope of this class)
"""

# ╔═╡ 93d48c9f-ec35-4f7b-9fcb-ab1d7f4c67d8
@section "Smooth Robustness"

# ╔═╡ 5e35b008-be58-4b31-9079-d2c5c8ed39af
begin
	q1 = [[1., 3., 6.],
		  [7., 1., 2.]]
	q2 = [[6., 1., 3.],
		  [5., 2., 3.]]
	q3 = [[2., 7., 1.],
		  [6., 2., 2.]]
	q4 = [[4., 3., 4.],
	      [4., 4., 3.]]
	q5 = [[1., 4., 5.],
		  [1., 8., 1.]]
end;

# ╔═╡ 3afb60e5-e8a1-4318-95aa-2d575d2749d6
Markdown.parse("""
Query 1: \$a = $(q1[1])\$
		
 \$\\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ b=$(q1[2])\$
""")

# ╔═╡ 2ffdc718-0dcf-4d0d-b29f-5d9c13baf184
Markdown.parse("""
Query 2: \$a = $(q2[1])\$
		
 \$\\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ b=$(q2[2])\$
""")

# ╔═╡ edc03e51-8c15-4a2c-95e5-d96bc92a3063
Markdown.parse("""
Query 3: \$a = $(q3[1])\$
		
 \$\\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ b=$(q3[2])\$
""")

# ╔═╡ 95463860-4c32-4850-8837-c505ed08ac6b
Markdown.parse("""
Query 4: \$a = $(q4[1])\$
		
 \$\\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ b=$(q4[2])\$
""")

# ╔═╡ 4f9476b9-cd6d-47b9-9a66-30f01bff2723
Markdown.parse("""
Query 5: \$a = $(q5[1])\$
		
 \$\\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ b=$(q5[2])\$
""")

# ╔═╡ 39b9a784-2c8b-46a2-a414-1252638ade67
begin
	Bonds = PlutoUI.BuiltinsNotebook.AbstractPlutoDingetjes.Bonds

	struct DarkModeIndicator
		default::Bool
	end
	
	DarkModeIndicator(; default::Bool=false) = DarkModeIndicator(default)

	function Base.show(io::IO, ::MIME"text/html", link::DarkModeIndicator)
		print(io, """
			<span>
			<script>
				const span = currentScript.parentElement
				span.value = window.matchMedia('(prefers-color-scheme: dark)').matches
			</script>
			</span>
		""")
	end

	Base.get(checkbox::DarkModeIndicator) = checkbox.default
	Bonds.initial_value(b::DarkModeIndicator) = b.default
	Bonds.possible_values(b::DarkModeIndicator) = [false, true]
	Bonds.validate_value(b::DarkModeIndicator, val) = val isa Bool

	md"> _Backend_"
end

# ╔═╡ 4bf458ec-529e-45a5-a258-d799b44f7638
md"""
 $\alpha$: $(@bind α Slider(0:0.025:1, show_value=true, default=0.7))
"""

# ╔═╡ 94faddc1-5e2d-4fd8-ad8f-9746e90b5dc3
begin
	dist = Beta(8, 2)
	xs = collect(range(0, 2000, length=201))
	ys = pdf.(dist, xs ./ 2000)

	function estimate_cvar(dist, α)
		var = quantile(dist, α)
		trunc_dist = Truncated(dist, var, Inf)
		return var, mean(rand(trunc_dist, 10000))
	end

	var, cvar = estimate_cvar(dist, α)
	
	prisk = plot(xs, ys, color=theblue, lw=3, grid=false, bg="transparent", background_color_inside="#1A1A1A", fg="white", xlims=(0, 2000), ylims=(0, 4), label=false, xlabel="Loss of Separation (m)")
	plot!(prisk, x->pdf(dist, x/2000), 2000 * var, 2000, lw=0, fillrange=0, color=0.5*thepurple, label=false)
	vline!(prisk, [0.8 * 2000], linestyle=:dash, color=theblue, label="Expected Value")
	vline!(prisk, [var * 2000], color=thepurple, label="VaR")
	vline!(prisk, [cvar * 2000], linestyle=:dash, color=thepurple, label="CVaR")
end

# ╔═╡ ad2c11f8-df02-4029-b696-5b95daf8e84a
@bind q1res Select(["no response", "a", "b"])

# ╔═╡ 3f5b97f5-cfc7-4498-86f7-57aa14e2d240
@bind q2res Select(["no response", "a", "b"])

# ╔═╡ 5713e4c1-2ef6-4a68-a87a-b9099ff2a4ce
@bind q3res Select(["no response", "a", "b"])

# ╔═╡ bcdff5b0-520d-4b72-ab62-c9421eb69d79
@bind q4res Select(["no response", "a", "b"])

# ╔═╡ c5cb5bf4-ffb5-45b6-80b0-b28200e6463b
@bind q5res Select(["no response", "a", "b"])

# ╔═╡ 564035c8-b6fd-47f9-be7c-dac779702652
begin
	init = HalfSpace(-[1.0, 1.0], -1.0)
	qs = [q1, q2, q3, q4, q5]
	res = [q1res, q2res, q3res, q4res, q5res]

	p = plot(init, xlims=(0,1), ylims=(0,1), aspect_ratio=:equal, grid=false, bg="transparent", background_color_inside="#1A1A1A", fg="white", alpha=0.8, color=theblue, xlabel="\$w_1\$", ylabel="\$w_2\$")
	for (query, res) in zip(qs, res)
		if res != "no response"
			hs = get_halfspace(query, res)
			plot!(p, hs, alpha=0.8, c=theblue)
		end
	end
	p
end

# ╔═╡ c6795b34-8cb8-4df3-92b7-0a1eef071fa9
md"""
 $w$: $(@bind w Slider(0:0.5:10, show_value=true, default=0))
"""

# ╔═╡ 10c171c7-d94e-4f04-81fb-05cb218e8c50
begin
	times = collect(1:10)
	τ = [-1.0, -3.2, 2.0, 1.5, 3.0, 0.5, -0.5, -2.0, -4.0, -1.5]

	ψ = @formula ◊(xₜ -> xₜ > 0)
	gradρ = ∇ρ̃(τ, ψ, w=w)[:]
	
	p1 = plot(times, τ, legend=false, grid=false, bg="transparent", background_color_inside="#1A1A1A", fg="white", xlabel="Time (s)", ylabel="s", xlims=(1, 10), ylims=(-5, 4), c=:lightgray, lw=2, markershape=:circle)
	hline!(p1, [ρ̃(τ, ψ, w=w)], c=theblue, linestyle=:dash, lw=1.0)
	
	p2 = scatter(times, gradρ, legend=false, grid=false, bg="transparent", background_color_inside="#1A1A1A", fg="white", xlabel="Time (s)", ylabel="s", xlims=(1, 10), ylims=(-0.2, 1.2), c=theblue, markershape=:circle)
	for (t, v) in zip(times, gradρ)
		plot!(p2, [t, t], [0, v], c=theblue, lw=1)
	end
	hline!(p2, [0], c=:gray, linestyle=:dash, lw=0.5)

	plot(p1, p2, layout=(2, 1))
end

# ╔═╡ 06edfbee-5d16-4e83-a4f9-caf5bd901c00
@bind dark_mode DarkModeIndicator()

# ╔═╡ Cell order:
# ╟─9f76c16b-6564-4614-b362-1b4a4804c89e
# ╟─f7456e13-d5e8-4064-a138-e3c77ab1e106
# ╟─1cfcbf7c-fa09-4755-bc3a-e3666a7fb1cb
# ╟─4bf458ec-529e-45a5-a258-d799b44f7638
# ╟─94faddc1-5e2d-4fd8-ad8f-9746e90b5dc3
# ╟─5d81cee1-9eb8-4592-9aae-ecb79b8d106f
# ╟─a4848c23-17c3-42f0-87c9-13e9cdeb47bc
# ╟─564035c8-b6fd-47f9-be7c-dac779702652
# ╟─3afb60e5-e8a1-4318-95aa-2d575d2749d6
# ╟─ad2c11f8-df02-4029-b696-5b95daf8e84a
# ╟─2ffdc718-0dcf-4d0d-b29f-5d9c13baf184
# ╟─3f5b97f5-cfc7-4498-86f7-57aa14e2d240
# ╟─edc03e51-8c15-4a2c-95e5-d96bc92a3063
# ╟─5713e4c1-2ef6-4a68-a87a-b9099ff2a4ce
# ╟─95463860-4c32-4850-8837-c505ed08ac6b
# ╟─bcdff5b0-520d-4b72-ab62-c9421eb69d79
# ╟─4f9476b9-cd6d-47b9-9a66-30f01bff2723
# ╟─c5cb5bf4-ffb5-45b6-80b0-b28200e6463b
# ╟─748f98c8-ed69-4021-925e-a59641b3115e
# ╟─e45364dd-ecaf-4b49-b98e-13f6670be244
# ╟─93d48c9f-ec35-4f7b-9fcb-ab1d7f4c67d8
# ╟─c6795b34-8cb8-4df3-92b7-0a1eef071fa9
# ╟─10c171c7-d94e-4f04-81fb-05cb218e8c50
# ╟─5e35b008-be58-4b31-9079-d2c5c8ed39af
# ╟─39b9a784-2c8b-46a2-a414-1252638ade67
# ╟─480491fb-9f19-49ee-8f0f-0bd8c1c352d8
# ╟─06edfbee-5d16-4e83-a4f9-caf5bd901c00
