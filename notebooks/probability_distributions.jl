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
	using Optim
	import Optim: minimizer, optimize
	using Turing
	using PlutoPapers

	default(fontfamily="Computer Modern", framestyle=:box) # LaTeX-style plotting
	theblue = RGB(128 / 255, 185 / 255, 255 / 255)
	thepurple = RGB(195 / 255, 184 / 255, 255 / 255)
	nothing
end

# ╔═╡ 362ef981-6e9c-4702-8a1f-4bf0a5110693
begin
	presentation = PlutoPaper(
		documentclass=Tufte(),
		title="Algorithms for Validation: System Modeling",
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

# ╔═╡ f5f81464-9211-4044-be69-a015c496241d
title(presentation)

# ╔═╡ 953b2537-c2ae-40ed-b514-58f4877d0fe7
@section "Gaussian Distribution"

# ╔═╡ 0f3e473e-9b79-4186-969f-1eb8e6f420b3
md"""
We can find a Gaussian distribution that fits data set 1 well, but we cannot find a Gaussian distribution does not fit data set 2 well because data set 2 is _multimodal_.
"""

# ╔═╡ 2f3a2b6a-ed97-4c9d-b8e4-407aafea0239
@section "Gaussian Mixture"

# ╔═╡ 81a7430c-fe4c-4772-b0f6-125112e1671f
md"""
Gaussian mixture models allow us to represent multimodel distributions. Below is the probability density of a mixture of two Gaussian distributions:

$p(x) = w_1 \mathcal N(x \mid \mu_1, \sigma_1^2) + (1-w_1)\mathcal N(x \mid \mu_2, σ_2^2)$
"""

# ╔═╡ 29e8c71f-3607-4201-a653-0dfcc84143d8
@section "Transforming Distributions"

# ╔═╡ 2009f6fc-25c5-4688-b4c5-db751812ff88
md"""
We can also transform simple distributions into more complex distributions. Consider the following simple Gaussian distribution:

$Z \sim \mathcal N(0, 1^2)$

The following code draws $10,000$ samples from it:
"""

# ╔═╡ e686652c-b021-48a1-b304-f61e97a3233b
zs = rand(Normal(0, 1), 10000);

# ╔═╡ d518ed94-6e0f-4288-987e-ba0844b3da16
md"""
Suppose we apply a function $f$ to $Z$ to get a new variable $X$. In this case, let's apply the cube root function:
"""

# ╔═╡ baff5c79-2999-4645-9806-62f85f201265
f(z) = cbrt(z);

# ╔═╡ 9a3c42db-e6eb-4698-8026-2c30c095a26a
md"""
We can plot a histogram over the samples and see that they form a new distribution:
"""

# ╔═╡ 3d852daa-eb49-4431-b221-a3f31e97c8db
@subsection "What is the probability density of this new distribution?"

# ╔═╡ 27775f2b-aa18-4273-9001-2af92b25555b
md"""
$Z \sim p_z(\cdot), \ \ X \sim p_x(\cdot)$
$x = f(z)$
If $f$ is **invertible** and **differentiable**, then

$p_x(x) = p_z(g(x))|g^\prime(x)|$
where $g$ is the inverse of $f$.
"""

# ╔═╡ e1a9cb05-7648-490e-83a7-aec0b03e9908
g(x) = x^3;

# ╔═╡ aedeac75-ac4a-47d3-b890-e28552290ae8
g′(x) = 3x^2;

# ╔═╡ 82e6228a-9817-412d-8a55-ba76a68a772a
pₓ(x) = pdf(Normal(0, 1), g(x)) * abs(g′(x));

# ╔═╡ c6891caa-6bcd-4a19-94e5-7b8fdeda135f
@subsection "We just transformed a unimodal distribution into a multimodal one! 😎"

# ╔═╡ 247df8f6-99d4-4f19-9c30-95dac7d3b4ce
md"""
- This type of transformation is quite powerful and is the key idea behind _normalizing flows_. 
- Normalizing flows use a series of parameterized differentiable and invertible functions to transform simple distributions into complex ones. 
- If you are interested in learning more about normalizing flows, I found [this tutorial](https://youtu.be/u3vVyFVU_lI?si=eIS2QXB2ItWAyr_z) to be quite helpful. (They are outside the scope of this class.)
- Below is a cool illustration of a normalizing flow (courtesy of Liam Kruse)
"""

# ╔═╡ 531676ed-a0e9-42d9-8599-5fb995fc30c1
md"""
> **Note**: To calculate the density, we required that the function be both **differentiable** and **invertible**.

We can still tranform simple distributions using functions that do not meet these criteria. We just may not be able to calculate their probability density. However, we can still draw samples! The plot below transforms a uniform distribution.
"""

# ╔═╡ 091d8e8e-1ceb-4f7e-867e-96ccd6a23d41
@section "Multivariate Gaussian Distribution"

# ╔═╡ bdbf60e0-439b-4b04-96a2-296c9110a8c8
@section "Fun Aside"

# ╔═╡ f29e71ad-87ac-452a-855a-d4162b81d9e0
md"""
What is the probability that the total of the number of wisdom teeth the four authors of _Algorithms for Validation_ have ever had is $9$?
"""

# ╔═╡ 0b74fda9-ed43-456c-b5cb-dd198718455a
md"""
Vector of probabilities of having each possible number of teeth (first entry is probability of having $0$ teeth, second entry is probability of having $1$, etc.) Based on a few minutes of googling and guessing and asking chatGPT -- take these with a grain of salt 🙂
"""

# ╔═╡ bc53124a-b6d7-4c0c-97ad-63c648de09dd
teeth_probs = [0.15, 0.025, 0.025, 0.025, 0.764, 0.01, 0.001];

# ╔═╡ db4900b6-a00d-4157-a9e2-283bda448b9c
md"""
Compute probability by:
1. Finding all possible combinations of four numbers between 0 and 7 that add up to the number of teeth
2. Summing the product of the values of each combination
"""

# ╔═╡ 15367aa6-0070-47be-86a9-bcc4e1ba5db8
function get_prob(nteeth)
	possibilities = [collect(vals).+1 for vals in Iterators.product(0:6, 0:6, 0:6, 0:6) if sum(vals)==nteeth]
	return sum([prod(teeth_probs[inds])] for inds in possibilities)[1]
end

# ╔═╡ a420a942-6417-47af-afb1-6e2c0dc7c5e3
Markdown.parse("""
Probability of having 9 total wisdom teeth among the four authors: $(round(get_prob(9), digits=2))
""")

# ╔═╡ 70f91db0-19aa-4b64-b4ff-28e8e69c6b1f
md"""
The full distribution over probabilities is below:
"""

# ╔═╡ 97d25df2-ee32-4473-a425-0998fa1ebfc6
begin
	ns = collect(1:18)
	prob_ns = [get_prob(n) for n in ns]
	bar(ns, prob_ns, grid=false, bg="transparent", background_color_inside="#1A1A1A", fg="white", legend=false, c=theblue, ylims=(0, 0.4), xlabel="Number of Wisdom Teeth of Four Book Authors", ylabel="Probability")
end

# ╔═╡ b4d98a5f-d0ea-43c2-b710-071b1fc3c866
@section "Parameter Learning Example"

# ╔═╡ 55b0daed-3bb8-4253-a042-46ec661bbf98
md"""
Suppose we have the following dataset of states and corresponding observations (sensor measurements):
"""

# ╔═╡ dec2928e-087a-409a-9ffd-1564baf177e6
@section "Maximum Likelihood Parameter Learning"

# ╔═╡ c537c8bf-2885-4b3e-8822-c62cdb486c69
@section "Bayesian Parameter Learning"

# ╔═╡ ac7ef821-4666-469a-bdf4-e938a550b5b1
@section "Flipping Frisbees"

# ╔═╡ 64911e61-b8de-4965-9991-96e2e46fb0ff
prior = Beta(1, 1);

# ╔═╡ fa23f953-4449-4896-9385-36d4598bddc0
begin
	Random.seed!(4)
	actual_dist = Normal(0.28, 0.79)
	samples = rand(actual_dist, 10000)

	Random.seed!(4)
	mixture_dist = MixtureModel([Normal(2, 1.3), Normal(-0.75, 0.5)], [0.65, 0.35])
	samples_mixture = rand(mixture_dist, 10000)

	agent = ProportionalController([-15., -8.])
	env = InvertedPendulum()
	sensor = AdditiveNoiseSensor(MvNormal(zeros(2), [0.05^2 0; 0 0.2^2]))
	inverted_pendulum = System(agent, env, sensor)
	ψ = LTLSpecification(@formula □(s -> abs(s[1]) < π / 4))

	Random.seed!(1)
	τs = [rollout(inverted_pendulum, d=41) for _ in 1:100]
	s = [step.s[1] for τ in τs for step in τ]
	o = [step.o[1] for τ in τs for step in τ]

	md"> _Data Set Creation_"
end

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

	nbsp = html"&nbsp"

	Div = PlutoUI.ExperimentalLayout.Div
	divcenter = Dict("display"=>"flex", "justify-content"=>"center")
	centered(content) = Div(content; style=divcenter)

	struct MaximumLikelihoodParameterEstimation
	    likelihood # p(y) = likelihood(x; θ)
	    optimizer  # optimization algorithm: θ = optimizer(f)
	end
	
	function fit(alg::MaximumLikelihoodParameterEstimation, data)
	    f(θ) = sum(-logpdf(alg.likelihood(x, θ), y) for (x,y) in data)
	    return alg.optimizer(f)
	end

	md"> _Backend_"
end

# ╔═╡ d2f4cda7-8e48-4872-be62-6845d7b6e7bf
@bind dset Select(["data set 1", "data set 2"])

# ╔═╡ 8379d554-9c4e-42f5-83ef-671a7ae193aa
md"""
show likelihood: $(@bind show_like CheckBox())
"""

# ╔═╡ 98ee4ce4-44cc-47d3-b5f0-7b483865a630
begin
	function plot_samples_v_dist(samples, dist; xlims=(-4, 4), ylims=(0, 0.6), cbins=theblue, cdist=:magenta, nbins=35, show_likelihood=show_like)
		b_range = range(xlims[1], xlims[2], length=nbins)
		p = histogram(samples, normalize=:pdf, legend=false, xlims=xlims, ylims=ylims, c=cbins, grid=false, bg="transparent", background_color_inside="#1A1A1A", fg="white", xlabel="\$x\$", ylabel="\$p(x)\$", bins=b_range)
		xs = collect(range(xlims[1], xlims[2], length=150))
		ys = pdf.(dist, xs)
		plot!(p, xs, ys, lw=5, c=cdist)
		if show_like
			ll = sum(logpdf(dist, sample) for sample in samples)
			title!(p, "\$\\log p(D \\mid \\theta) = \$ $(round(ll, digits =2))")
		end
		return p
	end

	function plot_samples_v_pdf(samples, dist; xlims=(-4, 4), ylims=(0, 0.6), cbins=theblue, cdist=:magenta, nbins=35)
		b_range = range(xlims[1], xlims[2], length=nbins)
		p = histogram(samples, normalize=:pdf, legend=false, xlims=xlims, ylims=ylims, c=cbins, grid=false, bg="transparent", background_color_inside="#1A1A1A", fg="white", xlabel="\$x\$", ylabel="\$p(x)\$", bins=b_range)
		xs = collect(range(xlims[1], xlims[2], length=150))
		ys = dist.(xs)
		plot!(p, xs, ys, lw=5, c=cdist)
		return p
	end

	function plot_samples(samples; xlims=(-4, 4), ylims=(0, 0.6), cbins=theblue, nbins=35)
		b_range = range(xlims[1], xlims[2], length=nbins)
		p = histogram(samples, normalize=:pdf, legend=false, xlims=xlims, ylims=ylims, c=cbins, grid=false, bg="transparent", background_color_inside="#1A1A1A", fg="white", xlabel="\$x\$", ylabel="\$p(x)\$", bins=b_range)
		return p
	end

	md"> _Plotting Code_"
end

# ╔═╡ 4b02fcb8-3db8-4e05-8a7c-c1dabad8a487
plot_samples_v_dist(zs, Normal())

# ╔═╡ 934a618e-c62e-45fb-814e-8840202e2997
md"""
 $\mu$: $(@bind μ Slider(-4:0.1:4, show_value=true, default=0))

 $σ$: $(@bind σ Slider(0.5:0.05:1.5, show_value=true, default=1.0))
"""

# ╔═╡ 4ca27a08-df52-4e2f-86ad-f79c84b93ccf
begin
	if dset == "data set 1"
		plot_samples_v_dist(samples, Normal(μ, σ))
	else
		plot_samples_v_dist(samples_mixture, Normal(μ, σ))
	end
end

# ╔═╡ afe2c330-f1d4-4893-8709-f629ed797eb0
md"""

 $\mu_1$: $(@bind μ1 Slider(-4:0.1:4, show_value=true, default=0))

 $\mu_2$: $(@bind μ2 Slider(-4:0.1:4, show_value=true, default=0))

 $\sigma_1$: $(@bind σ1 Slider(0.5:0.05:1.5, show_value=true, default=1.0))

 $\sigma_2$: $(@bind σ2 Slider(0.5:0.05:1.5, show_value=true, default=1.0))

 w1 $(@bind w Slider(0.0:0.05:1.0, default=0.5)) w2
"""

# ╔═╡ 70a51ee9-eb4e-487d-addd-fb03e588ff8e
begin
	if dset == "data set 1"
		plot_samples_v_dist(samples, MixtureModel([Normal(μ1, σ1), Normal(μ2, σ2)], [1-w, w]))
	else
		plot_samples_v_dist(samples_mixture, MixtureModel([Normal(μ1, σ1), Normal(μ2, σ2)], [1-w, w]))
	end
end

# ╔═╡ 74dfa650-15d7-4f7d-ad27-11d75b9f5b0b
centered(LocalResource(joinpath(@__DIR__, "..", "media", "normalizing_flows.gif")))

# ╔═╡ b92aa8c8-29c8-4ea4-89fb-73cd4f466346
centered(LocalResource(joinpath(@__DIR__, "..", "media", "dist_transform.svg")))

# ╔═╡ 28b11a32-f2dd-45af-8cbc-78f5b689c16a
md"""
 $\Sigma_x$: $(@bind Σx Slider(0.5:0.05:1.5, show_value=true, default=1.0))

 $\Sigma_y$: $(@bind Σy Slider(0.5:0.05:1.5, show_value=true, default=1.0))

 $\Sigma_{xy}$: $(@bind Σxy Slider(-1:0.1:1, show_value=true, default=0.0))
"""

# ╔═╡ 3319cfe2-7411-4adc-b9a6-c73e3464fe96
Markdown.parse("""
\$\\Sigma = \\begin{bmatrix} $(Σx) & $(Σxy) \\\\ $(Σxy) & $(Σy) \\end{bmatrix}\$
""")

# ╔═╡ 22a94e1b-8137-4388-8743-817017778f6a
begin
	Σ = [Σx Σxy; Σxy Σy]
	dist = MvNormal(zeros(2), Σ)
	f(x, y) = pdf(dist, [x, y])
	x = collect(range(-3, 3, length=200))
	y = collect(range(-3, 3, length=200))
	z = @. f(x', y)
	contour(x, y, z, cbar=false, grid=false, bg="transparent", background_color_inside=:black, fg="white", aspect_ratio=:equal, xlims=(-3, 3), ylims=(-3, 3), color=cgrad([:black, theblue]), lw=3, ticks=:none, xlabel="\$x\$", ylabel="\$y\$")
end

# ╔═╡ d054898c-04c7-4bb7-8b5c-d25130072dd4
xs = f.(zs);

# ╔═╡ 5316d2cf-ef2a-471e-a7ec-30d7c9889196
plot_samples(xs, cbins=thepurple, ylims=(0.0, 1.0))

# ╔═╡ 63be3472-3b52-4440-a43b-ddaab866dbee
plot_samples_v_pdf(xs, pₓ, cbins=thepurple, cdist=:magenta, ylims=(0.0, 1.0))

# ╔═╡ e6f9a59e-79f6-4a5e-ae2b-947d6eb93f8f
md"""
 $\theta_1$: $(@bind θ₁ Slider(-2.0:0.1:2.0, show_value=true, default=0.0))

 $\theta_2$: $(@bind θ₂ Slider(-2.0:0.1:2.0, show_value=true, default=0.0))

 $\theta_3$: $(@bind θ₃ Slider(0.01:0.01:0.2, show_value=true, default=0.1))
"""

# ╔═╡ aad58f2a-0e03-459d-af47-5f059489f9e3
begin
	f2(x, y) = pdf(Normal(θ₁ * x + θ₂, θ₃), y)
	xsmap = collect(range(-0.2, 0.2, length=201))
	ysmap = collect(range(-0.2, 0.2, length=201))
	zsmap = @. f2(xsmap', ysmap)
	p = heatmap(xsmap, ysmap, zsmap, cbar=false, grid=false, bg="transparent", background_color_inside=:black, fg="white", aspect_ratio=:equal, xlims=(-0.2, 0.2), ylims=(-0.2, 0.2), color=cgrad([:black, theblue]), alpha=0.75)
	llike = sum(logpdf(Normal(θ₁ * x + θ₂, θ₃), y) for (x, y) in zip(s, o))
	title!(p, "Log likelihood: $(round(llike, digits=2))")
	scatter!(p, s, o, c=:pink, bg="transparent", background_color_inside="#1A1A1A", fg="white", legend=false, aspect_ratio=:equal, xlims=(-0.2,0.2), ylims=(-0.2,0.2), xlabel="\$s\$", ylabel="\$o\$", markerstrokecolor=:pink, markersize=2, markeralpha=0.5)
end

# ╔═╡ 34208203-81b3-430e-8dfd-2fb6e5f81746
begin
	likelihood(x, θ) = Normal(θ[1] * x + θ[2], exp(θ[3]))
	optimizer(f) = minimizer(optimize(f, zeros(3), Optim.GradientDescent()))
	data = zip(s, o)

	Random.seed!(4)
	alg = MaximumLikelihoodParameterEstimation(likelihood, optimizer)
	θ = fit(alg, data)
	θ = [θ[1], θ[2], exp(θ[3])]

	# Print the results
	Markdown.parse("""
	\$p(y \\mid x) = \\mathcal N ($(round(θ[1], digits=3))x + $(abs(round(θ[2], digits=3))), $(round(θ[3], digits=2))^2)\$
	""")
end

# ╔═╡ b7c1bbe1-cb15-423e-8d3c-eae2ba42947e
begin
	f3(x, y) = pdf(Normal(θ[1] * x + θ[2], θ[3]), y)
	zsmap3 = @. f3(xsmap', ysmap)
	p3 = heatmap(xsmap, ysmap, zsmap3, cbar=false, grid=false, bg="transparent", background_color_inside=:black, fg="white", aspect_ratio=:equal, xlims=(-0.2, 0.2), ylims=(-0.2, 0.2), color=cgrad([:black, theblue]), alpha=0.75)
	llike3 = sum(logpdf(Normal(θ[1] * x + θ[2], θ[3]), y) for (x, y) in zip(s, o))
	title!(p3, "Log likelihood: $(round(llike3, digits=2))")
	scatter!(p3, s, o, c=:pink, bg="transparent", background_color_inside="#1A1A1A", fg="white", legend=false, aspect_ratio=:equal, xlims=(-0.2,0.2), ylims=(-0.2,0.2), xlabel="\$s\$", ylabel="\$o\$", markerstrokecolor=:pink, markersize=2, markeralpha=0.5)
end

# ╔═╡ 98038b85-e990-47f3-99fc-7c7397e07df4
md"""
Number of data points: $(@bind npoints NumberField(0:20:200, default=20))
"""

# ╔═╡ 01c52339-ebff-49f7-a34e-882270c33baa
begin
	struct BayesianParameterEstimation
		likelihood # p(y) = likelihood(x, θ)
		prior      # prior distribution
		sampler    # Turing.jl sampler
		m          # number of samples from posterior
	end
	
	
	function fit_bayesian(alg::BayesianParameterEstimation, data)
		x, y = first.(data), last.(data)
		@model function posterior(x, y)
			θ ~ alg.prior
			for i in eachindex(x)
				y[i] ~ alg.likelihood(x[i], θ)
			end
		end
		return Turing.sample(posterior(x, y), alg.sampler, alg.m)
	end
	
	inds = randperm(length(first.(data)))[1:npoints]
end;

# ╔═╡ cdd2bf01-f951-4a47-8bc2-64611fe0a89f
begin
	Random.seed!(4)
	prior_dist = MvNormal(zeros(3), 4 * I)
	sampler = NUTS()
	m = 1000
	
	alg_bayesian = BayesianParameterEstimation(likelihood, prior_dist, sampler, m)
	θdist = fit_bayesian(alg_bayesian, zip(first.(data)[inds], last.(data)[inds]))
end;

# ╔═╡ 87443c26-f162-41dc-a99a-a33a22d30f6b
begin
	function plot_data(data, var; xlims)
		p = histogram(data, normalize=:pdf, legend=false, c=theblue, grid=false, bg="transparent", background_color_inside="#1A1A1A", fg="white", xlabel="\$$var\$", ylabel="\$p($var)\$", size=(400, 400), title="Expected Value: $(round(mean(data), digits=2))", xlims=xlims)
		return p
	end
	
	pa = plot_data(θdist[Symbol("θ[1]")].data[:], "\\theta_1", xlims=(0,2))
	pb = plot_data(θdist[Symbol("θ[2]")].data[:], "\\theta_2", xlims=(-0.1,0.1))
	pc = plot_data(exp.(θdist[Symbol("θ[3]")].data[:]), "\\theta_3", xlims=(0,0.1))
	plot(pa, pb, pc, layout=(1, 3), size=(1200, 400))
end

# ╔═╡ 1adefa8e-faeb-4783-88d3-45be4ffa5a4b
md"""
Number same: $(@bind nsame NumberField(0:1:50, default=0))
Number different: $(@bind ndiff NumberField(0:1:50, default=0))

Posterior distribution:
"""

# ╔═╡ 8acdb942-28ec-4f28-ad31-1fc98ce09283
posterior = Beta(prior.α + nsame, prior.β + ndiff);

# ╔═╡ 24cf8fad-2496-46ac-8ecb-1dd51b5cc561
begin
	xspost = collect(range(0, 1, length=201))
	yspost = pdf.(posterior, xspost)
	plot(xspost, yspost, xlims=(0,1), ylims=(0,maximum(yspost)+0.1), legend=false,
	lw=2, c=theblue, grid=false, bg="transparent", background_color_inside="#1A1A1A", fg="white", xlabel="\$\\theta\$", ylabel="\$P(\\theta \\mid D\$", size=(300, 300), title="Beta($(Int(posterior.α)), $(Int(posterior.β)))")
end

# ╔═╡ 06edfbee-5d16-4e83-a4f9-caf5bd901c00
@bind dark_mode DarkModeIndicator()

# ╔═╡ 6ef48df6-aba6-4c7e-887a-4d6d60349627
toc()

# ╔═╡ Cell order:
# ╟─362ef981-6e9c-4702-8a1f-4bf0a5110693
# ╟─f5f81464-9211-4044-be69-a015c496241d
# ╟─953b2537-c2ae-40ed-b514-58f4877d0fe7
# ╟─d2f4cda7-8e48-4872-be62-6845d7b6e7bf
# ╟─8379d554-9c4e-42f5-83ef-671a7ae193aa
# ╟─934a618e-c62e-45fb-814e-8840202e2997
# ╟─4ca27a08-df52-4e2f-86ad-f79c84b93ccf
# ╟─0f3e473e-9b79-4186-969f-1eb8e6f420b3
# ╟─2f3a2b6a-ed97-4c9d-b8e4-407aafea0239
# ╟─81a7430c-fe4c-4772-b0f6-125112e1671f
# ╟─afe2c330-f1d4-4893-8709-f629ed797eb0
# ╟─70a51ee9-eb4e-487d-addd-fb03e588ff8e
# ╟─29e8c71f-3607-4201-a653-0dfcc84143d8
# ╟─2009f6fc-25c5-4688-b4c5-db751812ff88
# ╠═e686652c-b021-48a1-b304-f61e97a3233b
# ╟─4b02fcb8-3db8-4e05-8a7c-c1dabad8a487
# ╟─d518ed94-6e0f-4288-987e-ba0844b3da16
# ╠═baff5c79-2999-4645-9806-62f85f201265
# ╠═d054898c-04c7-4bb7-8b5c-d25130072dd4
# ╟─9a3c42db-e6eb-4698-8026-2c30c095a26a
# ╟─5316d2cf-ef2a-471e-a7ec-30d7c9889196
# ╟─3d852daa-eb49-4431-b221-a3f31e97c8db
# ╟─27775f2b-aa18-4273-9001-2af92b25555b
# ╠═e1a9cb05-7648-490e-83a7-aec0b03e9908
# ╠═aedeac75-ac4a-47d3-b890-e28552290ae8
# ╠═82e6228a-9817-412d-8a55-ba76a68a772a
# ╟─63be3472-3b52-4440-a43b-ddaab866dbee
# ╟─c6891caa-6bcd-4a19-94e5-7b8fdeda135f
# ╟─247df8f6-99d4-4f19-9c30-95dac7d3b4ce
# ╟─74dfa650-15d7-4f7d-ad27-11d75b9f5b0b
# ╟─531676ed-a0e9-42d9-8599-5fb995fc30c1
# ╟─b92aa8c8-29c8-4ea4-89fb-73cd4f466346
# ╟─091d8e8e-1ceb-4f7e-867e-96ccd6a23d41
# ╟─28b11a32-f2dd-45af-8cbc-78f5b689c16a
# ╟─3319cfe2-7411-4adc-b9a6-c73e3464fe96
# ╟─22a94e1b-8137-4388-8743-817017778f6a
# ╟─bdbf60e0-439b-4b04-96a2-296c9110a8c8
# ╟─f29e71ad-87ac-452a-855a-d4162b81d9e0
# ╟─0b74fda9-ed43-456c-b5cb-dd198718455a
# ╠═bc53124a-b6d7-4c0c-97ad-63c648de09dd
# ╟─db4900b6-a00d-4157-a9e2-283bda448b9c
# ╠═15367aa6-0070-47be-86a9-bcc4e1ba5db8
# ╟─a420a942-6417-47af-afb1-6e2c0dc7c5e3
# ╟─70f91db0-19aa-4b64-b4ff-28e8e69c6b1f
# ╟─97d25df2-ee32-4473-a425-0998fa1ebfc6
# ╟─b4d98a5f-d0ea-43c2-b710-071b1fc3c866
# ╟─55b0daed-3bb8-4253-a042-46ec661bbf98
# ╟─e6f9a59e-79f6-4a5e-ae2b-947d6eb93f8f
# ╟─aad58f2a-0e03-459d-af47-5f059489f9e3
# ╟─dec2928e-087a-409a-9ffd-1564baf177e6
# ╠═34208203-81b3-430e-8dfd-2fb6e5f81746
# ╟─b7c1bbe1-cb15-423e-8d3c-eae2ba42947e
# ╟─c537c8bf-2885-4b3e-8822-c62cdb486c69
# ╟─98038b85-e990-47f3-99fc-7c7397e07df4
# ╠═cdd2bf01-f951-4a47-8bc2-64611fe0a89f
# ╟─87443c26-f162-41dc-a99a-a33a22d30f6b
# ╟─01c52339-ebff-49f7-a34e-882270c33baa
# ╟─ac7ef821-4666-469a-bdf4-e938a550b5b1
# ╠═64911e61-b8de-4965-9991-96e2e46fb0ff
# ╟─1adefa8e-faeb-4783-88d3-45be4ffa5a4b
# ╠═8acdb942-28ec-4f28-ad31-1fc98ce09283
# ╟─24cf8fad-2496-46ac-8ecb-1dd51b5cc561
# ╟─fa23f953-4449-4896-9385-36d4598bddc0
# ╟─98ee4ce4-44cc-47d3-b5f0-7b483865a630
# ╟─39b9a784-2c8b-46a2-a414-1252638ade67
# ╟─480491fb-9f19-49ee-8f0f-0bd8c1c352d8
# ╟─06edfbee-5d16-4e83-a4f9-caf5bd901c00
# ╟─6ef48df6-aba6-4c7e-887a-4d6d60349627
