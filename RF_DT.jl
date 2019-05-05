using DataFrames
using CSV
using Statistics
using Random
using StatsBase

include("lib/gini_impurity.jl")
include("lib/helpers.jl")

# Show how data - nodes are structed
mutable struct Node
    data::DataFrame
    index::Int64
    gini::Float64
    isLeaf::Bool
    way::String
    mean::Any
    feature::Int64
    nodeTrue::Node
    nodeFalse::Node
    Node(data,index,gini,isLeaf,way,mean,feature) = new(data,index,gini,isLeaf,way,mean,feature)
end

# Show how data - trees are structed
mutable struct Tree
    nodes::Array{Node}
    result::String
    n_cols::Int64
    order_cols::Array{Int64}
    Tree() = new()
end

function train_test(S::DataFrame, test_size::Float64 = 0.25)
    train = S[trunc(Int, floor( size(S,1) * test_size ) ) + 1 : size(S,1),:]
    test  = S[1:trunc(Int, floor( size(S,1) * test_size ) ),:]
    return train, test
end

function random_train_test(S::DataFrame, test_size::Float64 = 0.25)
    train = S[1: trunc(Int, floor( size(S,1) * test_size )) + 1, :]
    test  = S[trunc(Int, floor( size(S,1) * test_size )) + 2:size(S,1), :]
    return train, test
end

function random_feature(S::DataFrame, n_max_features::Integer = 2)
    # Random select some features
    random_feat = shuffle(1:size(S,2)-1)[1:n_max_features]
    # Insert the last feature = Y columns
    push!(random_feat, size(S,2))
    # Return the random selected itens
    return random_feat
end

function build_random_ds(S::DataFrame, size_sample::Int64 = 50)
    data_samples = sample([i for i in range(1, length=size(S,1))],size_sample)
    return S[data_samples, :]
end

function getMin(ItemDict, Index)
    min     = Inf
    ret_min = (Integer,Any,Float64)
    for ItemArray in ItemDict
        for Item in ItemArray[2]
            if Item[Index] < min
                min = Item[Index]
                ret_min = Item
            end
        end        
    end
    return ret_min
end

function getMaxOccur(items)
    max   = 0
    occur = ""
    for (key, value) in items
        if value > max
            occur = key
            max = value
        end
    end
    return occur
end

function BuildTree(S::DataFrame, NodeFrom::Node, Nodes::Array{Node}, Position::Integer = 0, GiniImpurity::Float64 = 1.0, Way::String = "Root")
    # Get the node
    features_impurity = Dict(
        "String" => Array{Tuple{Integer,String,Float64}}(undef,0),
        "Float64" => Array{Tuple{Integer,Float64,Float64}}(undef,0)
    )
    for j in range(1,length=size(S,2)-1)        
        if (eltypes(S)[j].b == String)
            mean_imp, gini_imp = gini_impurity_string(S, j, GiniImpurity)
            push!(features_impurity["String"], (j, mean_imp, gini_imp))
        else
            mean_imp, gini_imp = gini_impurity(S, j, GiniImpurity)
            push!(features_impurity["Float64"], (j, mean_imp, gini_imp))
        end
    end
    node_min = getMin(features_impurity,3)
    
    if (size( unique(S[:,size(S,2)]) , 1 ) > 1) && (size( unique( S[:,node_min[1]] ), 1) > 1)
        node = Node(S, Position, node_min[3], false, Way, node_min[2], node_min[1])
        # Go to left - true
        BuildTree(filterDS(S, true, node_min), node, Nodes, Position + 1, 1., "True")
        # Go to right - false
        BuildTree(filterDS(S, false, node_min), node, Nodes, Position + 1, 1., "False")
    else
        node = Node(S, Position, node_min[3], true, Way, node_min[2], node_min[1])
    end
    if Way == "True"
        NodeFrom.nodeTrue = node
    end
    if Way == "False"
        NodeFrom.nodeFalse = node
    end

    push!(Nodes, node)    
end

function Accuracy(predictions::Array{Tuple{Any,Any,Bool}})
    perc_right_answer = size(filter(x -> x[:][3], predictions), 1) / size(predictions,1)
    return perc_right_answer * 100
end

function Predict(Test::DataFrame, predictions::Array{Tuple{Any,Any,Bool}}, nodes::Array{Node})
    result_node = (Any, Any, Bool)
    for row in eachrow(Test)
        node = getRoot(nodes)
        while !node.isLeaf
            if check_predict(row[node.feature], node.mean)
                # True
                node = node.nodeTrue
            else
                # False
                node = node.nodeFalse
            end
        end
        y_value = row[:][size(Test,2)]
        if node.isLeaf
            result_node = (y_value, unique(node.data[:,size(Test,2)])[1], unique(node.data[:,size(Test,2)])[1] == y_value)
            push!(predictions, result_node)
        end
    end
end

function PredictTrees(Test::DataFrame, trees::Array{Tree}, predictions::Array{Tuple{Any,Any,Bool}})
    
    for row in eachrow(Test)
        predictions_in_place = Dict(y => 0 for y in unique(Test[:,size(Test,2)]))
        for tree in trees
            node = getRoot(tree.nodes)
            # Put the row on the right order from train tree
            row_pred = row[tree.order_cols]
            while !node.isLeaf
                if check_predict(row_pred[node.feature], node.mean)
                    # True
                    node = node.nodeTrue
                else
                    # False
                    node = node.nodeFalse
                end
            end
            if node.isLeaf
                if haskey(predictions_in_place, unique(node.data[:,tree.n_cols])[1])
                    predictions_in_place[unique(node.data[:,tree.n_cols])[1]] += 1
                else
                    push!(predictions_in_place, (unique(node.data[:,tree.n_cols])[1] => 1))
                end
            end
        end
        y_value = row[:][size(Test,2)]
        max_variety = getMaxOccur(predictions_in_place)
        result_node = (y_value, max_variety, max_variety == y_value)
        
        push!(predictions, result_node)
    end               
    
end

function getRoot(Nodes::Array{Node})
    for node in Nodes
        if node.way == "Root"
            return node
        end
    end
end

function decision_tree(S::DataFrame, S_Test::DataFrame)
    # Start result of predictions
    predictions = Array{Tuple{Any,Any,Bool}}(undef,0)
    
    # Start the root node of tree and array of Nodes
    node_root   = Node(S, 0, 0.,false, "None", 0., 0)
    nodes       = Array{Node}(undef,0)
    
    # Start building and trainig the tree
    BuildTree(S, node_root, nodes, 0, 1.0)
    
    # Make predictions
    Predict(S_Test, predictions, nodes)

    return predictions
end

function random_forest(S::DataFrame, S_Test::DataFrame, n_trees::Int64 = 100, n_max_features::Int64 = 2)
    predictions = Array{Tuple{Any,Any,Bool}}(undef,0)
    trees       = Array{Tree}(undef, 0)
    
    for i in 1:n_trees
        
        # Build random DS
        S_train_local = build_random_ds(S,size(S,1))

        # Filter DS with random selected features
        random_feat = random_feature(S_train_local, n_max_features)
        S_in_place  = S_train_local[:,random_feat]
        
        # Nodes for starter
        nodes       = Array{Node}(undef,0)
        node_root   = Node(S_in_place, 0, 0.,false, "None", 0., 0)

        # Start building and trainig the tree
        BuildTree(S_in_place, node_root, nodes, 0, 1.0)
        
        tree = Tree()
        tree.nodes = nodes
        tree.n_cols     = size(S_in_place, 2)
        tree.order_cols = random_feat
        push!(trees, tree)

    end
    # Do the predictions
    PredictTrees(S_Test, trees, predictions)  
    
    # Return result of predictions
    return predictions
end

# Random dataset
S_dataset   = CSV.read("data/abalone.data")
# Split the data into train_dataset and test_dataset
train, test = train_test(S_dataset, 0.25)

# Result of predictions
predictions_rf = random_forest(train, test, 1, size(train,2)-1)
# Get accuracy of random Forest
accu_1 = Accuracy(predictions_rf)
@show "Resultado com RF"
@show accu_1

# Do the predictions
#predictions_dt = decision_tree(train, test)
# Get accuraty
#accu_2 = Accuracy(predictions_dt)
#@show "Resultado com DT"
#@show accu_2