<div align="center">
    <img src=".github/Rango_logo.png" height="150"/>
    <h1>Rango</h1>
    <img src="https://circle.circleci.sbg.intuit.com/gh/asharma19/Rango/tree/master.svg?style=svg" alt="Rango-CircleCI-Status"/>
    <img src="https://img.shields.io/badge/contributions-welcome-orange" />
</div>

A machine learning model which classifies the documents based on pre-trained as well as trainable word embeddings in Julia. It also provides with a simple document matcher to match your documents semantically.

## Structure
The repository consists of 5 source files.

`dependencies.jl`  This is the julia script for installing all the dependency packages required for the model to work.
`ModelPreTrained.jl`  This provides the deep learning model with pretrained [GloVe](https://nlp.stanford.edu/projects/glove/) embeddings to use for document classification.
`ModelManual.jl`  This provides the deep learning model with self trainable embeddings for document classification.
`MatchDocs.jl`  This leverages simple cosine distance between word/sentence vectors to match documents.

## Usage
Well, assuming you have [Julia](https://julialang.org/downloads/) installed in your system/workspace, you need to have a few Julia packages.
If you are running the code first time you need to install all the dependency packages. The dependencies script will help you download the packages required for the model to run.

```julia
julia src/dependencies.jl
```

Alternatively, you can use docker and pull in the [julia-nlp image](https://hub.docker.com/repository/docker/akaysh/julia-nlp) I created on docker hub.

### Pre-Trained Embeddings

After installing the packages you can use the [ModelPreTrained.jl](src/ModelPreTrained.jl) script to train/use the deep learning model with GloVe embeddings.

```
Model Specifications - Pretrained GloVe Embedding layer of 50 dimensions, LSTM cell of dims (max_features X 100), a hidden layer (relu activation) with dimensions (100 X 50) and an output layer of dimensions (50 x number of labels).
```

```julia
include("src/ModelPreTrained.jl")
model, test_data = train()
@save "/models/GlovModel.bson" model
```
To load back the trained model and use it for testing:

```julia
@load "/models/GlovModel.bson" model
Flux.reset!(model)
args = Args()
train_data, test_data, doc_term_matrix, klasses = get_processed_data(args)
test(model,test_data)
```

#### Keyword arguments
You can pass Keyword arguments in the `train` function.

- `lr`: Learning rate for the model to learn. This is passed in a gradient descent optimiser for the weights.
- `τ`: Max iterations/epochs for the training.
- `path` Path to the data file which contains train and test data.
- `max_length` Max length of the docs allowed to be used to form vectors which are fed into the LSTM. Sentences greater than the length are trimmed and sentences lesser than the length are padded.

### Trainable Embeddings
```julia
include("src/ModelManual.jl")
@save "/models/EmbModel.bson" m
```
To load back the trained model and use it for testing:

```julia
@load "/models/EmbModel.bson" model
args = Args()
train_data, test_data, doc_term_matrix, klasses = get_processed_data(args)
test(model,test_data)
```

### Document Matcher

You can use Document matcher to match and return top 20 documents/sentences from a given data to your sentence. Save and load for faster results. It uses cosine distance between sentence vectors with the help of pre trained embeddings.
```julia
include("src/MatchDocs.jl")
match_docs(args)
```
#### keyword arguments

`path`: path to the data for sentences  
`string`: string to be matched

If you are running this for the first time the `match_docs` function will save the sentences and vectors in data folder as `sent_vec.csv` and `sentences.csv`. For faster results you can use `load_and_match` after the vectors have been saved.

```julia
load_and_match(sentence_path, sent_vecs_path, string)
```

## Tutorials
There are a few tutorials provided in the [Tutorials](tutorials) folder which show examples of document classification using the above model.

For details on how to contribute to this project see [CONTRIBUTING.md](.github/CONTRIBUTING.md)
