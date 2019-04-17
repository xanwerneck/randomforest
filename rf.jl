using DataFrames
using CSV
using Statistics
using Random

include("lib/gini_impurity.jl")
include("lib/helpers.jl")

include("dt.jl")

# Show how data - trees are structed
mutable struct Tree
    nodes::Array{Node}
    result::String
    n_cols::Int64
    order_cols::Array{Int64}
    Tree() = new()
end
predictions = Array{Tuple{String,String,Bool}}(undef,0)
trees       = Array{Tree}(undef, 0)

function PredictTrees(Test)
    
    for row in eachrow(Test)
        predictions_in_place = Dict(y => 0 for y in unique(Test[:,size(Test,2)]))
        for tree in trees
            node = GetRootRF(tree.nodes)
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
        max_variety = GetMaxOccur(predictions_in_place)
        result_node = (y_value, max_variety, max_variety == y_value)
        
        push!(predictions, result_node)
    end               
    
end

function BuildTreeRandomForest(S, NodeFrom, Nodes, Position = 0, GiniImpurity = 1.0, Way = "Root")
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
    node_min = GetMinRF(features_impurity,3)
    
    if (size( unique(S[:,size(S,2)]) , 1 ) > 1) && (size( unique( S[:,node_min[1]] ), 1) > 1)
        node = Node(S, Position, node_min[3], false, Way, node_min[2], node_min[1])
        # Go to left - true
        BuildTreeRandomForest(filterDS(S, true, node_min), node, Nodes, Position + 1, 1., "True")
        # Go to right - false
        BuildTreeRandomForest(filterDS(S, false, node_min), node, Nodes, Position + 1, 1., "False")
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

function TrainRandomForest(train, number_trees, number_max_features)
    for i in 1:number_trees
        N, M = size(train)
        
        # Build a sample from train dataset
        S_train_local = train[rand(1:N, N),:]

        # Get a subset of sample with number of features
        random_feat = [i for i in 1:M]
        if number_max_features < M
            random_feat = RandomFeature(S_train_local, number_max_features)            
        end 
        S_in_place  = S_train_local[:,random_feat]       

        # Build the tree
        nodes       = Array{Node}(undef,0)
        node_root   = Node(S_in_place, 0, 0.,false, "None", 0., 0)

        # Start building and trainig the tree
        BuildTreeRandomForest(train, node_root, nodes, 0, 1.0)
        
        tree = Tree()
        tree.nodes = nodes
        tree.n_cols     = size(S_in_place, 2)
        tree.order_cols = random_feat
        
        push!(trees, tree)
    end
end

function TestRandomForest(test)
    PredictTrees(test)
end