using  Languages, TextAnalysis, Flux, PyPlot, Statistics, MLDataUtils, Embeddings, MLLabelUtils
using Flux: logitcrossentropy, normalise, onecold, onehotbatch
using Parameters: @with_kw
using BSON: @save,@load
@with_kw mutable struct Args
    lr::Float64 = 0.01 # learning rate
    τ::Int = 110 # max iterations
    path::String = "Rango/data/sample.txt" # path to data
    max_length::Int = 4 # Max length of the doc to be considered
    max_features::Int = 8 #Number of features per word to learn for training
end
# Padding the corpus wrt the longest document
function pad_corpus(c, pad_size, word_dict)
    M = []
    for doc in 1:length(c)
        tks = tokens(c[doc])
        if length(tks) >= pad_size
            tk_indexes = [tk_idx(w,word_dict) for w in tks[1:pad_size]]
        end
        if length(tks) < pad_size
            tk_indexes = zeros(Int64,pad_size - length(tks))
            tk_indexes = vcat(tk_indexes, [tk_idx(w,word_dict) for w in tks])
        end
        doc == 1 ? M = tk_indexes' : M = vcat(M, tk_indexes')
    end
    return M
end
# function to return the index of the word in the word dictionary
tk_idx(s,word_dict) = haskey(word_dict, s) ? i = word_dict[s] : i = 0
function get_processed_data(args)
    f = open(args.path)
    doc_array = readlines(f)
    labels, texts = [], []
    for doc in doc_array
        content = split(doc)
        push!(labels,content[1])
        push!(texts,join(content[2:end]," "))
    end
    # pushing the text from the files to the string documents
    docs = []
    for i in 1:length(texts)
        push!(docs, StringDocument(texts[i]))
    end
    # building a Corpus
    corpus = Corpus(docs)
    # updating the lexicon and creating the word dict
    update_lexicon!(corpus)
    doc_term_matrix = DocumentTermMatrix(corpus)
    word_dict = doc_term_matrix.column_indices
    # splitting words in the document
    word_docs = map(s -> split(s,r"[,. ]",keepempty=false),texts)
    # pad size is the number of words in the maximum word document
    # Can set a fixed length or the max doc length
    pad_size = args.max_length
    # padding the docs
    padded_docs = pad_corpus(corpus, pad_size,word_dict)
    # forming the data with the labels
    X = padded_docs'
    (X_train, y_train), (X_test, y_test) = splitobs((X, labels); at = 0.67)
    klasses = sort(unique(labels))
    y_train = convert(Array{Bool}, y_train .== "True")
    y_test = convert(Array{Bool}, y_test .== "True")
    y_train = reshape(y_train,1,size(y_train,1))
    y_test = reshape(y_test,1,size(y_test,1))
    train_data = [(X_train, y_train)]
    test_data = [(X_test, y_test)]
    return train_data, test_data, doc_term_matrix, klasses
end
# Function to build confusion matrix
function confusion_matrix(X, y, model)
    ŷ = onehotbatch(onecold(model(X)), 1:2)
    y * transpose(ŷ)
end
function getArgs(; kws...)
    # Initialize hyperparameter arguments
    global args = Args(; kws...)
end
function test(model, test)
    # Testing model performance on test data
    X_test, y_test = test[1]
    accuracy_score = accuracy(X_test, y_test, model)
    println("\nAccuracy: $accuracy_score")
    # Sanity check. TO be done in testing
    # @assert accuracy_score > 0.6
    println("\nConfusion Matrix:\n")
    display(confusion_matrix(X_test, y_test, model))
end
getArgs()
#Loading processed data
train_data, test_data, doc_term_matrix, klasses = get_processed_data(args)
word_dict = doc_term_matrix.column_indices
X_train, y_train = train_data[1]
max_features = args.max_features
pad_size = args.max_length
# Building Flux Embeddings
N = size(X_train,2)  #Number of documents
# features per word to learn, depends on the size of the corpus, larger corpus will probably need a higher dimension
# number of words in the vocabulary, should always be higher than the maximum index in our dictionary.
vocab_size = maximum(word_dict)[2] + 1
# Embedding layer for Flux model
# glorot_normal returns an Array of size dims containing random variables taken from a normal distribution with mean 0 and standard deviation (2 / sum(dims)).
struct EmbeddingLayer
   W
   EmbeddingLayer(mf, vs) = new(Flux.glorot_normal(mf, vs))
end
# Enabling Flux to learn
@Flux.treelike EmbeddingLayer
(m::EmbeddingLayer)(x) = m.W * Flux.onehotbatch(reshape(x, pad_size*N), 0:vocab_size-1)
m = Chain(EmbeddingLayer(max_features, vocab_size),
          x -> reshape(x, max_features, pad_size, N),
          x -> sum(x, dims=2),
          x -> reshape(x, max_features, N),
          Dense(max_features, 1, σ)
)
# Defining loss function to be used in training
# Training
# Gradient descent optimiser with learning rate `args.lr`
loss_h = []
accuracy_train = []
accuracy(x, y) = mean(x .== y)
loss(x, y) = sum(Flux.binarycrossentropy.(m(x), y))
optimizer = Flux.Descent(args.lr)
println("Starting training.")
for epoch in 1:args.τ
    Flux.train!(loss, Flux.params(m), train_data, optimizer)
    println(loss(X_train, y_train), " ", accuracy(m(X_train).>0.5,y_train))
    push!(loss_h, loss(X_train, y_train))
    push!(accuracy_train, accuracy(m(X_train).>0.5,y_train))
end
