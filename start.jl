
include("rf.jl")

function main_rf(file)
    # Empty array of trees
    trees = Array{Tree}(undef, 0)

    # Empty array of predictions
    predictions = Array{Tuple{Any,Any,Bool}}(undef,0)

    # Read dataset file
    S_dataset   = CSV.read(file)
    # Build 2 dataframe - train with 75% and test with 25%
    train, test = TrainTest(S_dataset, 0.25)

    # Train and build a single tree
    # Input: train : dataset of train
    # number_trees : number of trees
    # number_columns : number of random columns selected
    TrainRandomForest(train,1,size(train,2),trees)
    
    # Predict the test dataset
    PredictTrees(test, predictions, trees)

    # Get accuracy
    accu = Accuracy(predictions)
    @show accu   
end


main_rf(string("data/", ARGS[1]))