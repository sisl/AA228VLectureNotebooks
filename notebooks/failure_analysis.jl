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
	using PlutoPapers

	default(fontfamily="Computer Modern", framestyle=:box) # LaTeX-style plotting
end;

# ╔═╡ 2f308228-2806-4bdf-b7df-e000c6eb277a
begin
	presentation = PlutoPaper(
		documentclass=Tufte(),
		title="Algorithms for Validation: Failure Analysis",
		authors=[
			# Author(name="Lecture Introduction")
			# Author(name="Mykel Kochenderfer")
			# Author(name="Sydney Katz")
			# Author(name="Anthony Corso")
			# Author(name="Robert Moss")
		]
	)
	
	applyclass(presentation.documentclass)
end

# ╔═╡ 19fb47bd-f479-4842-ad51-6f1af88c72f8
title(presentation)

# ╔═╡ 763e2587-81de-43b0-970b-511f7bdb48ba
@section "Problem Parameters"

# ╔═╡ 974588a3-00ac-45da-84fb-bcccb7027f11
@section "Falsification and Failure Distribution"

# ╔═╡ 7d6362a9-71a0-4926-8571-416dcc6cdec9
@section "Failure Probability Estimate"

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

# ╔═╡ 934a618e-c62e-45fb-814e-8840202e2997
md"""
Perception noise: $(@bind σ Slider(0.01:0.01:0.3, show_value=true, default=0.25))

Number of rollouts: $(@bind m Slider(1:1:150, show_value=true, default=50))
"""

# ╔═╡ 06edfbee-5d16-4e83-a4f9-caf5bd901c00
@bind dark_mode DarkModeIndicator()

# ╔═╡ 98ee4ce4-44cc-47d3-b5f0-7b483865a630
begin
	sys = System(
		ProportionalController([-15.0, -8.0]),
		InvertedPendulum(),
		AdditiveNoiseSensor(MvNormal(zeros(2), σ^2*I))
	)

	ψ = LTLSpecification(@formula □(s -> abs(s[1]) < π / 4))

	simulate(m) = [rollout(sys, d=41) for i in 1:m]

	function set_aspect_ratio!(p)
		x_range = xlims()[2] - xlims()[1]
		y_range = ylims()[2] - ylims()[1]
		plot!(p, ratio=x_range/y_range)
	end
		
	rectangle(w, h, x, y) = Shape(x .+ [0,w,w,0], y .+ [0,0,h,h])
	
	function plot_it(sys, ψ, τ=missing;
					is_dark_mode=dark_mode,
					title="Inverted Pendulum",
					max_lines=100, size=(680,350), plot_successes=true, kwargs...)
		if is_dark_mode
			p = plot(
				size=size,
				grid=false,
				bg="transparent",
				background_color_inside="#1A1A1A",
				fg="white",
			)
		else
			p = plot(
				size=size,
				grid=false,
				bg="transparent",
				background_color_inside="white",
			)
		end
	
		plot!(p, rectangle(2, 1, 0, π/4), opacity=0.5, color="#F5615C", label=false)
		plot!(p, rectangle(2, 1, 0, -π/4-1), opacity=0.5, color="#F5615C", label=false)
		xlabel!(p, "Time (s)")
		ylabel!(p, "𝜃 (rad)")
		title!(p, title)
		xlims!(p, 0, 2)
		ylims!(p, -1.2, 1.2)
		set_aspect_ratio!(p)
	
		function plot_pendulum_traj!(p, τ; lw=2, α=1, color="#009E73")
			X = range(0, step=sys.env.dt, length=length(τ))
			plot!(p, X, [step.s[1] for step in τ]; lw, color, α, label=false)
		end
	
		if τ isa Vector{<:Vector}
			# Multiple trajectories
			τ_successes = filter(τᵢ->!isfailure(ψ, τᵢ), τ)
			τ_failures = filter(τᵢ->isfailure(ψ, τᵢ), τ)
			if plot_successes
				for (i,τᵢ) in enumerate(τ_successes)
					if i > max_lines
						break
					else
						plot_pendulum_traj!(p, τᵢ; lw=1, α=0.75, color="#009E73")
					end
				end
			end
	
			for τᵢ in τ_failures
				plot_pendulum_traj!(p, τᵢ; lw=1, α=1, color="#F5615C")
			end
		elseif τ isa Vector
			# Single trajectory
			get_color(ψ, τ) = isfailure(ψ, τ) ? "#F5615C" : "#009E73"
			plot_pendulum_traj!(p, τ; lw=2, color=get_color(ψ, τ))
		end
	
		return p
	end

	function plot_both(sys, ψ, τ=missing;
					is_dark_mode=dark_mode,
					title="Inverted Pendulum",
					max_lines=100, size=(680,350))
		p1 = plot_it(sys, ψ, τ, is_dark_mode=dark_mode, title="Falsification", max_lines=max_lines, size=size)
		p2 = plot_it(sys, ψ, τ, is_dark_mode=dark_mode, title="Failure Distribution", max_lines=max_lines, size=size, plot_successes=false)
		return plot(p1, p2)
	end

	md"> _Plotting Code_"
end

# ╔═╡ c06bda0b-8340-418b-a872-97c183bd865f
begin
	Random.seed!(0)
	τs = simulate(m)
	nfail = sum(isfailure(ψ, τ) for τ in τs)
	failurestring = nfail == 1 ? "failure" : "failures"
	trajstring = m == 1 ? "trajectory" : "trajectories"

	md"> _Simulation Code_"
end

# ╔═╡ fc54b8e1-e992-41d7-a7e7-383d29a35fcc
plot_both(sys, ψ, τs; max_lines=500)

# ╔═╡ 8070b914-ad92-4cdf-b611-85f488d6ebda
Markdown.parse("""
\$P_\\text{fail} = \\frac{$nfail \\text{\\ \\ $failurestring}}{$m \\text{\\ \\ $trajstring}} = $(round(nfail/m, digits=3))\$
""")

# ╔═╡ Cell order:
# ╟─2f308228-2806-4bdf-b7df-e000c6eb277a
# ╟─19fb47bd-f479-4842-ad51-6f1af88c72f8
# ╟─763e2587-81de-43b0-970b-511f7bdb48ba
# ╟─934a618e-c62e-45fb-814e-8840202e2997
# ╟─974588a3-00ac-45da-84fb-bcccb7027f11
# ╟─fc54b8e1-e992-41d7-a7e7-383d29a35fcc
# ╟─7d6362a9-71a0-4926-8571-416dcc6cdec9
# ╟─8070b914-ad92-4cdf-b611-85f488d6ebda
# ╟─98ee4ce4-44cc-47d3-b5f0-7b483865a630
# ╟─c06bda0b-8340-418b-a872-97c183bd865f
# ╟─39b9a784-2c8b-46a2-a414-1252638ade67
# ╟─480491fb-9f19-49ee-8f0f-0bd8c1c352d8
# ╟─06edfbee-5d16-4e83-a4f9-caf5bd901c00
