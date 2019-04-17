
include("rf.jl")

function main_rf(file)
    # Read dataset file
    S_dataset   = CSV.read(file)
    # Build 2 dataframe - train with 75% and test with 25%
    train, test = TrainTest(S_dataset, 0.25)

    # Train and build a single tree
    # Input: train : dataset of train
    # number_trees : number of trees
    # number_columns : number of random columns selected
    TrainRandomForest(train,100,4)
    
    # Predict the test dataset
    TestRandomForest(test)

    # Get accuraty
    accu = Accuracy()
    @show accu   
end

main_rf(string("data/", ARGS[1]))

function main_dt(file)
    # Read dataset file
    S_dataset   = CSV.read(file)
    # Build 2 dataframe - train with 75% and test with 25%
    train, test = TrainTest(S_dataset, 0.25)

    # Train and build a single tree
    # Input: train : dataset of train
    TrainTree(train)
    
    # Predict the test dataset
    TestTree(test)

    # Get accuraty
    accu = Accuracy()
    @show accu   
end

main_dt(string("data/", ARGS[1]))