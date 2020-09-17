#!/usr/bin/env julia
import Pkg;
println("Installing dependencies for the model.");
Pkg.add("Languages");
Pkg.add("TextAnalysis");
Pkg.add("BSON");
Pkg.add("Flux");
Pkg.add("PyPlot");
Pkg.add("Statistics");
Pkg.add("MLDataUtils");
Pkg.add("Embeddings")
Pkg.add("MLLabelUtils");
Pkg.add("Parameters");
println("Finished adding packages.");
