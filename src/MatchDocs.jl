using Distances, Statistics
using MultivariateStats
using PyPlot
using WordTokenizers
using TextAnalysis
using DelimitedFiles
using Embeddings
using Parameters: @with_kw
@with_kw mutable struct Args
    path::String = "data/pg345.txt" # path to data
    string::String = "my favorite food is strawberry ice cream" # Max length of the doc to be considered
end
const embtable = load_embeddings(GloVe{:en},1) # or load_embeddings(FastText_Text) or ...
#Function to return the index of the word in the embedding (returns 0 if the word is not found)
const get_word_index = Dict(word=>ii for (ii,word) in enumerate(embtable.vocab))
embeddings = embtable.embeddings
vocab = embtable.vocab
vec_size, vocab_size = size(embeddings)
println("Loaded embeddings, each word is represented by a vector with $vec_size features. The vocab size is $vocab_size")
vec_idx(s) = findfirst(x -> x == s, vocab)
function vec(s)
    if vec_idx(s)!= nothing
        embeddings[:, vec_idx(s)]
    end
end
# To compare the similarity of document vectors
cosine(x,y) = 1 - cosine_dist(x, y)
function closest(v, n=20)
    list = [(x,cosine(embeddings'[x,:], v)) for x in 1:size(embeddings)[2]]
    topn_idx = sort(list, by = x -> x[2], rev = true)[1:n]
    return [vocab[a] for (a,_) in topn_idx]
end
# Function to get a mean sentence vector
function sentvec(s,sentences)
    local arr = []
    for w in split(sentences[s])
        if vec(w) != nothing
            push!(arr, vec(w))
        end
    end
    if length(arr) == 0
        ones(Float32, (50,1))*999
    else
        mean(arr)
    end
end
function closest_sent(input_str, n=20)
    mean_vec_input = mean([vec(w) for w in split(input_str)])
    list = [(x,cosine(mean_vec_input, sentvec(x))) for x in 1:length(sentences)]
    topn_idx=sort(list, by = x -> x[2], rev = true)[1:n]
    return [sentences[a] for (a,_) in topn_idx]
end
function match_docs(; kws...)
    args = Args(; kws...)
    txt = open(args.path) do file
        read(file, String)
    end
    println("Loaded $(args.path), length=$(length(txt)) characters")
    txt = replace(txt, r"\n|\r|_|,|—" => " ")
    txt = replace(txt, r"[\"""*();!]" => "")
    sd = StringDocument(txt)
    prepare!(sd, strip_whitespace)
    global sentences = split_sentences(sd.text)
    i = 1
    for s in 1:length(sentences)
        if length(split(sentences[s]))>3
            sentences[i]=lowercase(replace(sentences[s], "."=>""))
            i += 1
        end
    end
    save_vector(sentences)
    return closest_sent(args.string)
end
function save_vector(sentences)
    i = length(sentences) + 1
    sent_vecs = []
    for s in 1:length(sentences)
        i == 1 ? sent_vecs=sentvec(s) : push!(sent_vecs,sentvec(s))
    end
    writedlm( "data/sent_vec.csv",  sent_vecs, ',')
    writedlm( "data/sentences.csv",  sentences, ',')
end
function closest_sent_pretrained(pretrained_arr, input_str, n=20)
    mean_vec_input = mean([vec(w) for w in split(input_str)])
    list = [(x,cosine(mean_vec_input, pretrained_arr[x,:])) for x in 1:length(sentences)]
    topn_idx = sort(list, by = x -> x[2], rev=true)[1:n]
    return [sentences[a] for (a,_) in topn_idx]
end
function load_and_match(sentences_path, sent_vecs_path, string)
    sentences = readdlm("sentences.csv", '!', String, header=false)
    sent_vecs = readdlm("sent_vec.csv", ',', Float32, header=false)
    return closest_sent_pretrained(sent_vecs, string)
end
