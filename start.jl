
include("dt.jl")

function main(file)
    # Read dataset file
    S_dataset   = CSV.read(file)
    # Build 2 dataframe - train with 75% and test with 25%
    train, test = TrainTest(S_dataset, 0.25)
    # Train and build a single tree
    TrainTree(train)
    # Test results
    TestTree(test)
    # Get accuraty
    accu = Accuracy()
    @show accu   
end

main("data/dataframe.csv")