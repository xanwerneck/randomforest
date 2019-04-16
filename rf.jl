using DataFrames
using CSV
using Statistics
using Random

# Show how data - nodes are structed
mutable struct Node
    data::DataFrame
    index::Int64
    gini::Float64
    isLeaf::Bool
    way::String
    mean::Float64
    feature::Int64
    nodeTrue::Node
    nodeFalse::Node
    Node(data,index,gini,isLeaf,way,mean,feature) = new(data,index,gini,isLeaf,way,mean,feature)
end

# Show how data - trees are structed
mutable struct Tree
    nodes::Array{Node}
    result::String
    Tree() = new()
end

function train_test(S::DataFrame, test_size::Float64 = 0.25)
    train = S[trunc(Int, floor( size(S,1) * test_size ) ) + 1 : size(S,1),:]
    test  = S[1:trunc(Int, floor( size(S,1) * test_size ) ),:]
    return train, test
end

function random_feature(S::DataFrame, n_max_features::Integer = 2)
    # Random select some features
    random_feat = shuffle(1:size(S,2)-1)[1:n_max_features]
    # Insert the last feature = Y columns
    push!(random_feat, size(S,2))
end

function gini_impurity(S::DataFrame, Y_uniques::Array, Feature::Integer, S_imp::Float64, SizeDS)
    #Number of columns
    m = size(S,2)
    #Filter just one feature
    data_impurity = S[:,Feature]
    #Distinct values of dataframe
    unique_feature = sort(unique(data_impurity))

    #Means between intervals
    unique_means   = []    
    for i in range(1,length=size(unique_feature,1)-1)
        push!(unique_means, ( unique_feature[i] + unique_feature[i+1] ) / 2 )
    end
    
    #Minimun of impurity
    gini_impurity_feature = (0,size(unique_means, 1),0)
    for mean in unique_means

        node_left  = filter(x -> (x[:][Feature] <= mean), S)
        node_right = filter(x -> (x[:][Feature] > mean), S)
        
        gini_impurity_left = 1.0
        for y in range(1, length=size(Y_uniques, 1))
            gini_impurity_left -= ( count(x->(x==Y_uniques[y]),node_left[:,m]) / size(node_left,1) ) ^ 2
        end
                
        gini_impurity_right = 1.0
        for y in range(1, length=size(Y_uniques, 1))
            gini_impurity_right -= ( count(x->(x==Y_uniques[y]),node_right[:,m]) / size(node_right,1) ) ^ 2
        end

        gini_impurity_node = ( ( size(node_left,1) / size(S,1) ) * gini_impurity_left ) + ( ( size(node_right,1) / size(S,1) ) * gini_impurity_right )
        if (gini_impurity_node < gini_impurity_feature[2]) && (gini_impurity_feature[2] > 0.0)
            gini_impurity_feature = (mean, gini_impurity_node, Feature)
        end

    end
    return gini_impurity_feature
end

function getMax(ItemArray, Index)
    max     = -Inf
    ret_max = (0.,0.,0.)
    for item in ItemArray
        if item[Index] > max
            max = item[Index]
            ret_max = item
        end
    end
    return ret_max
end

function getMin(ItemArray, Index)
    min     = Inf
    ret_min = (1.,1.,1.)
    for item in ItemArray
        if item[Index] < min
            min = item[Index]
            ret_min = item
        end
    end
    return ret_min
end

function getMaxOccur(items)
    max   = 0
    occur = ""
    for item in items
        max_item = 0
        for item_contain in items
            if item == item_contain
                max_item = max_item + 1
            end
        end
        if max_item > max
            max = max_item
            occur = item
        end
    end
    return occur
end

function BuildTree(S::DataFrame, NodeFrom::Node, Nodes::Array{Node}, Position::Integer = 0, GiniImpurity::Float64 = 1.0, Way::String = "Root")
    # Get the node
    features_impurity = Array{Tuple{Integer,Float64,Float64}}(undef,0)
    imps = []
    for j in range(1,length=size(S,2)-1)
        mean_imp, gini_imp = gini_impurity(S, unique(S[:,size(S,2)]), j, GiniImpurity, size(S_dataset, 1))        
        push!(features_impurity, (j, mean_imp, gini_imp))
    end
    node_min = getMin(features_impurity,3)

    if size( unique(S[:,size(S,2)]) , 1 ) > 1
        node = Node(S, Position, node_min[3], false, Way, node_min[2], node_min[1])
        # Go to left - true
        BuildTree(filter(x -> x[:][node_min[1]] <= node_min[2],S), node, Nodes, Position + 1, 1., "True")
        # Go to right - false
        BuildTree(filter(x -> x[:][node_min[1]] > node_min[2],S), node, Nodes, Position + 1, 1., "False")
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

function Accuracy(predictions::Array{Tuple{String,String,Bool}})
    perc_right_answer = size(filter(x -> x[:][3], predictions), 1) / size(predictions,1)
    return perc_right_answer * 100
end

function PredictTrees(Test::DataFrame, trees::Array{Tree}, predictions::Array{Tuple{String,String,Bool}})
    for row in eachrow(Test)
        predictions_in_place = Array{String}(undef, 0)
        for tree in trees
            node = getRoot(tree.nodes)

            while !node.isLeaf
                if row[node.feature] <= node.mean
                    # True
                    node = node.nodeTrue
                else
                    # False
                    node = node.nodeFalse
                end
            end
            if node.isLeaf
                push!(predictions_in_place, unique(node.data[:,size(Test,2)])[1])
            end
        end
        max_variety = getMaxOccur(predictions_in_place)
        result_node = (row[:variety], max_variety, max_variety == row[:variety])
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

function random_forest(S::DataFrame, n_trees::Integer = 100, n_max_features::Integer = 3)
    predictions = Array{Tuple{String,String,Bool}}(undef,0)
    trees       = Array{Tree}(undef, 0)
    for i in 1:n_trees
        # Build random DS
        Escale_ds     = size(S,1)
        S_train_local = S[shuffle(1:size(S, 1)),:][1:Escale_ds, :]

        # Filter DS with random selected features
        random_feat = random_feature(S_train_local)
        S_in_place  = S_train_local[:,random_feat]

        # Nodes for starter
        nodes       = Array{Node}(undef,0)
        node_root   = Node(S_in_place, 0, 0.,false, "None", 0., 0)

        # Start building and trainig the tree
        BuildTree(train, node_root, nodes, 0, 1.0)
        
        tree = Tree()
        tree.nodes = nodes
        push!(trees, tree)

    end
    # Do the predictions
    PredictTrees(test, trees, predictions)   
    
    # Return result of predictions
    return predictions
end

# Random dataset
S_dataset   = CSV.read("dataframe.csv")
# Split the data into train_dataset and test_dataset
train, test = train_test(S_dataset)

# Result of predictions
predictions_rf = random_forest(S_dataset)
# Get accuracy of random Forest
accu = Accuracy(predictions_rf)
    
@show accu