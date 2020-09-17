# Testing Glov pretrained embedding with LSTM model
include("../src/ModelPreTrained.jl")
model, test_data = train()
Flux.reset!(model)
test(model,test_data)
