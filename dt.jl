using DataFrames
using CSV
using Statistics
using Random

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
    # Change the construct of struct
    Node(data,index,gini,isLeaf,way,mean,feature) = new(data,index,gini,isLeaf,way,mean,feature)
end
# Empty array of Nodes
nodes       = Array{Node}(undef,0)
# Empty array of predictions
predictions = Array{Tuple{String,String,Bool}}(undef,0)

function GImpurity(S, Y_uniques, Feature, S_imp, SizeDS)
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
    if size(unique_means, 1) == 0
        unique_means = unique_feature
    end
    
    #Minimun of impurity
    gini_impurity_feature = (0,size(unique_means, 1),0)
    for mean in unique_means

        node_left  = filter(x -> (x[:][Feature] <= mean), S)
        node_right = filter(x -> (x[:][Feature] > mean), S)
            
        gini_impurity_left = 1.0
        if size(node_left,1) > 0  
            for y in range(1, length=size(Y_uniques, 1))
                gini_impurity_left -= ( count(x->(x==Y_uniques[y]),node_left[:,m]) / size(node_left,1) ) ^ 2
            end
        end
                
        gini_impurity_right = 1.0
        if size(node_right,1) > 0  
            for y in range(1, length=size(Y_uniques, 1))
                gini_impurity_right -= ( count(x->(x==Y_uniques[y]),node_right[:,m]) / size(node_right,1) ) ^ 2
            end
        end

        gini_impurity_node = ( ( size(node_left,1) / size(S,1) ) * gini_impurity_left ) + ( ( size(node_right,1) / size(S,1) ) * gini_impurity_right )
        if (gini_impurity_node < gini_impurity_feature[2]) && (gini_impurity_feature[2] > 0.0)
            gini_impurity_feature = (mean, gini_impurity_node, Feature)
        end

    end
    return gini_impurity_feature
end

function GetMin(ItemArray, Index)
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

function BuildTree(S, NodeFrom, Position = 0, GiniImpurity = 1.0, Way = "Root")
    # Get the node
    features_impurity = Array{Tuple{Integer,Float64,Float64}}(undef,0)
    for j in range(1,length=size(S,2)-1)
        mean_imp, gini_imp = GImpurity(S, unique(S[:,size(S,2)]), j, GiniImpurity, size(S, 1))        
        push!(features_impurity, (j, mean_imp, gini_imp))
    end
    node_min = GetMin(features_impurity,3)
    
    if (size( unique(S[:,size(S,2)]) , 1 ) > 1) && (size( unique( S[:,node_min[1]] ), 1) > 1)
        node = Node(S, Position, node_min[3], false, Way, node_min[2], node_min[1])
        # Go to left - true
        BuildTree(filter(x -> x[:][node_min[1]] <= node_min[2],S), node, Position + 1, 1., "True")
        # Go to right - false
        BuildTree(filter(x -> x[:][node_min[1]] > node_min[2],S), node, Position + 1, 1., "False")
    else
        node = Node(S, Position, node_min[3], true, Way, node_min[2], node_min[1])
    end
    if Way == "True"
        NodeFrom.nodeTrue = node
    end
    if Way == "False"
        NodeFrom.nodeFalse = node
    end

    push!(nodes, node)    
end

function TrainTest(S, test_size = 0.25)
    train = S[trunc(Int, floor( size(S,1) * test_size ) ) + 1 : size(S,1),:]
    test  = S[1:trunc(Int, floor( size(S,1) * test_size ) ),:]
    return train, test
end

function Accuracy()
    perc_right_answer = size(filter(x -> x[:][3], predictions), 1) / size(predictions,1)
    return perc_right_answer * 100
end

function Predict(Test)
    Size_test   = size(Test,1)
    
    for row in eachrow(Test)
        node = GetRoot()
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
            result_node = (row[:variety], unique(node.data[:,size(Test,2)])[1], unique(node.data[:,size(Test,2)])[1] == row[:variety])
            push!(predictions, result_node)
        end
    end
end

function GetRoot()
    for node in nodes
        if node.way == "Root"
            return node
        end
    end
end

function TrainTree(train)
    # Create a pseudo root node
    node_root   = Node(train, 0, 0.,false, "None", 0., 0)
    # Start creating the tree
    BuildTree(train, node_root, 0, 1.0)
end

function TestTree(test)
    # Make predictions based on tree
    Predict(test)
end