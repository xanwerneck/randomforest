
include("dt.jl")

function main_dt(file)
    # Empty array of Nodes
    nodes       = Array{Node}(undef,0)

    # Empty array of predictions
    predictions = Array{Tuple{Any,Any,Bool}}(undef,0)

    # Read dataset file
    S_dataset   = CSV.read(file)[1:500,:]
    # Build 2 dataframe - train with 75% and test with 25%
    println("---- Split the data ----")
    train, test = TrainTest(S_dataset, 0.25)
    
    # Train and build a single tree
    # Input: train : dataset of train
    println("---- Train the data ----")
    TrainTree(train, nodes)
    
    # Predict the test dataset
    println("---- Predict the test data ----")
    Predict(test, predictions, nodes)

    # Get accuracy
    accu = Accuracy(predictions)
    @show accu   
end

main_dt(string("data/", ARGS[1]))