# Stanford AA228V/CS238V Lecture Notebooks
[![website](https://img.shields.io/badge/website-Stanford-b31b1b.svg)](https://aa228v.stanford.edu/)
[![textbook](https://img.shields.io/badge/textbook-MIT%20Press-0072B2.svg)](https://algorithmsbook.com/validation/)
[![package](https://img.shields.io/badge/package-StanfordAA228V.jl-175E54.svg)](https://github.com/sisl/StanfordAA228V.jl)

Notebooks to accompany lectures for Stanford's AA228V/CS238V _Validation of Safety-Critical Systems_.

<p align="center"> <img src="./media/coverart.svg"> </p>

Uses the [`StanfordAA228V.jl`](https://github.com/sisl/StanfordAA228V.jl) Julia package.

## Julia/Pluto Session
[![html](https://img.shields.io/badge/static%20html-Julia%20Pluto-0072B2)](https://sisl.github.io/AA228VLectureNotebooks/media/html/julia_pluto_session.html)
[![html](https://img.shields.io/badge/static%20html-Julia%20Plotting-0072B2)](https://sisl.github.io/AA228VLectureNotebooks/media/html/julia_plotting.html)

The session on using Julia and Pluto can be viewed as a [static notebook](https://sisl.github.io/AA228VLectureNotebooks/media/html/julia_pluto_session.html) or run locally by downloading [`notebooks/julia_pluto_session.jl`](https://github.com/sisl/AA228VLectureNotebooks/blob/main/notebooks/julia_pluto_session.jl) and opening it in a Pluto session. A link to the session recording will be on Canvas.


## Installation
- _Install [**Pluto**](https://plutojl.org/) and the SISL Julia package registry:_
    1. Open `julia` in a terminal.
    1. Run:
        ```julia
        using Pkg
        Pkg.add("Pluto")
        Registry.add(url="https://github.com/sisl/Registry.git")
        ```
## Running
- Open Pluto:
    - Run `julia` in a terminal.
    - Within Julia, run:
        ```julia
        using Pluto
        Pluto.run()
        ```
        - This should open Pluto in your browser.
            - If not, checkout the Pluto output for the URL (likely http://localhost:1234/):
            ```julia
            ┌ Info:
            └ Opening http://localhost:1234/ in your default browser... ~ have fun!
            ```
