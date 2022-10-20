using Flux
x = rand(8)
W = rand(4, 8) # weights
b = rand(4) # bias
layer₁(x) = 1.0 ./ (1.0.+exp.(-W*x - b)) # first layer, writing the function
layer₂(x) = σ.(W * x .+ b) # second layer, using the in-built sigmoid function
layer₃ = Dense(8,4,σ) # using the in-built dense function
layer₃(x)



# More layers in a model:
Layer₁ = Dense(10, 5, relu) # Activation function - relu
Layer₂ = Dense(5, 2)
Layer₃ = softmax

# chain() - connecting functions in Julia into chains
m = Chain(Layer₁ , Layer₂, Layer₃)

# defining the cost function
x, y = rand(10), rand(2)
loss(p, y) = sum((p.-y).^2)/length(y)
loss(m(x),y)

# using flux's cost function
Flux.mse(m(x),y)

# generalization error
# I need to regularize the model, so it can affectively approximate data other than the training one
using LinearAlgebra
L₁(θ) = sum(abs, θ)  # Lasso, L1
L₂(θ) = sum(abs2, θ)
J(x,y,W) = loss(m(x),y) + L₁(W)
J(x,y,W)
