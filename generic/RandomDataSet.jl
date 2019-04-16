using DataFrames
using CSV
using Random

function RandomData(file)
    S = CSV.read(file)    
    S = S[shuffle(1:size(S, 1)),:]
    S = S[shuffle(1:size(S, 1)),:]
    CSV.write("data/dataframe.csv", S[shuffle(1:size(S, 1)),:])
end

RandomData("data/iris.csv")