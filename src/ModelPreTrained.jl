using  Languages, TextAnalysis, Flux, PyPlot, Statistics, MLDataUtils, Embeddings, MLLabelUtils
using Flux: logitcrossentropy, normalise, onecold, onehotbatch
using Parameters: @with_kw
using BSON: @save,@load

@with_kw mutable struct Args
    lr::Float64 = 0.05 # learning rate
    τ::Int = 110 # max iterations
    path::String = "data/short-amazon-reviews.txt" # path to data
    max_length::Int = 4 # Max length of the doc to be considered
end

#Function to return the index of the word in the embedding (returns 0 if the word is not found)
function vec_idx(s,vocab)
    i=findfirst(x -> x==s, vocab)
    i==nothing ? i=0 : i
end

#This function provides the GloVe word vector of the given word.
function wvec(s,embeddings,vocab)
    return embeddings[:, vec_idx(s,vocab)]
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
tk_idx(s,word_dict) = haskey(word_dict, s) ? i = word_dict[s] : i=0


function load_embeddings()
    embtable = Embeddings.load_embeddings(GloVe{:en},1) # or load_embeddings(FastText_Text) or ...
    embeddings = embtable.embeddings
    vocab = embtable.vocab
    return vocab,embeddings
end

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
    word_docs = map(s -> split(s,r"[,. ]",keepempty = false),texts)
    # pad size is the number of words in the maximum word document

    # Can set a fixed length or the max doc length
    # pad_size = maximum(length(word_docs[i]) for i in 1:length(texts))
    pad_size = args.max_length

    # padding the docs
    padded_docs = pad_corpus(corpus, pad_size,word_dict)
    # forming the data with the labels
    X = padded_docs'

    (X_train, y_train), (X_test, y_test) = splitobs((X, labels); at = 0.67)

    klasses = sort(unique(labels))
    y_train = Flux.onehotbatch(y_train,klasses)
    y_test = Flux.onehotbatch(y_test,klasses)
    train_data = [(X_train, y_train)]
    test_data = [(X_test, y_test)]

    return train_data, test_data, doc_term_matrix, klasses
end

# Accuracy Function
accuracy(x, y, model) = mean(onecold(model(x)) .== onecold(y))

# Function to build confusion matrix
function confusion_matrix(X, y, model)
    ŷ = onehotbatch(onecold(model(X)), 1:2)
    y * transpose(ŷ)
end



function train(; kws...)
    # Initialize hyperparameter arguments
    args = Args(; kws...)

    #Loading processed data
    train_data, test_data, doc_term_matrix, klasses = get_processed_data(args)
    word_dict = doc_term_matrix.column_indices
    X_train, y_train = train_data[1]

    # loading the embeddings
    embedding_vocab, embeddings = load_embeddings()
    max_features,em_vocab_size = size(embeddings)

    # Building Flux Embeddings
    N = size(X_train,2)  #Number of documents
    # features per word to learn, depends on the size of the corpus, larger corpus will probably need a higher dimension
    # number of words in the vocabulary, should always be higher than the maximum index in our dictionary.
    vocab_size = maximum(word_dict)[2] + 1


    # Embedding layer for Flux model
    # glorot_normal returns an Array of size dims containing random variables taken from a normal distribution with mean 0 and standard deviation (2 / sum(dims)).
    embedding_matrix = Flux.glorot_normal(max_features, vocab_size)

    # Creating the embedding matrix
    for term in doc_term_matrix.terms
        if vec_idx(term,embedding_vocab)!= 0
            embedding_matrix[:,word_dict[term] + 1] = wvec(term,embeddings,embedding_vocab)
        end
    end

    m = Chain((x) -> (embedding_matrix * Flux.onehotbatch(reshape(x, args.max_length*size(x,2)), 0:vocab_size - 1),size(x,2)),
          (arg) -> (reshape(arg[1], max_features, args.max_length, arg[2]),arg[2]),
          (arg) -> (sum(arg[1], dims = 2),arg[2]),
          (arg) -> reshape(arg[1], max_features, arg[2]),
        LSTM(max_features,100),
        Dense(100,50),
        Dense(50, size(klasses,1), σ)
         )

    # Defining loss function to be used in training
    # For numerical stability, we use here logitcrossentropy
    loss(x, y) = sum(Flux.logitcrossentropy(m(x), y))

    # Training
    # Gradient descent optimiser with learning rate `args.lr`
    optimiser = Descent(args.lr)

    println("Starting training.")
    loss_h = []
    accuracy_train = []

    for epoch in 1:args.τ
        Flux.train!(loss, Flux.params(m), train_data, optimiser)

        println(loss(X_train, y_train), " ", accuracy(X_train, y_train, m))
        push!(loss_h, loss(X_train, y_train))
        push!(accuracy_train, accuracy(X_train, y_train, m))
    end

    print(Flux.onecold(m(X_train)), accuracy(X_train, y_train, m))

    return m, test_data
end

function test(model, test)
    # Testing model performance on test data
    X_test, y_test = test[1]
    accuracy_score = accuracy(X_test, y_test, model)

    println("\nAccuracy: $accuracy_score")

    # Sanity check.
    # @assert accuracy_score > 0.6

    println("\nConfusion Matrix:\n")
    display(confusion_matrix(X_test, y_test, model))
end
