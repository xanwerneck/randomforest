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
    mean::Any
    feature::Int64
    nodeTrue::Node
    nodeFalse::Node
    # Change the construct of struct
    Node(data,index,gini,isLeaf,way,mean,feature) = new(data,index,gini,isLeaf,way,mean,feature)
end

function ComputeEntropy(S, M)
    Y_uniques = unique(S[:,M])
    S_entropy = 0.0
    N         = size(S,1)

    for y in range(1, length=size(Y_uniques, 1))
        p_y = count(x->(x==Y_uniques[y]),S[:,M]) / N
        S_entropy += -(p_y * log(p_y))
    end
    
    return S_entropy
end

function InformationGain(S, S_left, S_right, M)
    Avg_left  = size(S_left,1) / size(S,1)
    Avg_right = size(S_right,1) / size(S,1)
    
    return ComputeEntropy(S,M) - (Avg_left * ComputeEntropy(S_left,M)) - (Avg_right * ComputeEntropy(S_right,M))
end

function ComputeInformationGain(S, Feature)
    #Number of columns
    N, M = size(S)
    #Filter just one feature
    data_impurity = S[:,Feature]
    
    #Distinct values of dataframe
    unique_values_feature = unique(data_impurity)
    
    #Minimun of impurity
    gain_information = (0.,0.,0)
    
    for value_feature in unique_values_feature

        S_left  = filter(x -> (x[:][Feature] <= value_feature), S)
        S_right = filter(x -> (x[:][Feature] > value_feature), S)
        
        gain = InformationGain(S, S_left, S_right, M)
        
        if (gain > gain_information[2])
            gain_information = (value_feature, gain, Feature)
        end
    end

    return gain_information
end

function GImpurity(S, Feature)
    #Number of columns
    N, M = size(S)
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
    gini_impurity_feature = (1.,size(unique_means, 1),0)
    for mean in unique_means

        node_left  = filter(x -> (x[:][Feature] <= mean), S)
        node_right = filter(x -> (x[:][Feature] > mean), S)
            
        gini_impurity_left = 1.0
        if size(node_left,1) > 0  
            Y_uniques_left = unique(node_left[:,M])
            for y in range(1, length=size(Y_uniques_left, 1))
                gini_impurity_left -= ( count(x->(x==Y_uniques_left[y]),node_left[:,M]) / size(node_left,1) ) ^ 2
            end
        end
                
        gini_impurity_right = 1.0
        if size(node_right,1) > 0  
            Y_uniques_right = unique(node_right[:,M])
            for y in range(1, length=size(Y_uniques_right, 1))
                gini_impurity_right -= ( count(x->(x==Y_uniques_right[y]),node_right[:,M]) / size(node_right,1) ) ^ 2
            end
        end

        gini_impurity_node = ( ( size(node_left,1) / N ) * gini_impurity_left ) + ( ( size(node_right,1) / N ) * gini_impurity_right )
        #gini_impurity_node = gini_impurity_left + gini_impurity_right
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

function GetMax(ItemArray, Index)
    max     = 0.
    ret_max = (0,0.,0.)
    for item in ItemArray
        if item[Index] > max
            max = item[Index]
            ret_max = item
        end
    end
    return ret_max
end

function BuildTree(S, NodeFrom, Nodes, Position = 0, Way = "Root")
    N, M = size(S)
    # Get the node
    features_impurity = Array{Tuple{Integer,Float64,Float64}}(undef,0)
    for j in range(1,length=M-1)
        mean_imp, gini_imp = ComputeInformationGain(S, j)        
        push!(features_impurity, (j, mean_imp, gini_imp))
    end
    node_min = GetMax(features_impurity,3)
    
    if (size( unique(S[:,M]) , 1 ) > 1) && (size( unique( S[:,node_min[1]] ), 1) > 1)
        node = Node(S, Position, node_min[3], false, Way, node_min[2], node_min[1])
        # Go to left - true
        BuildTree(filter(x -> x[:][node_min[1]] <= node_min[2],S), node, Nodes, Position + 1, "True")
        # Go to right - false
        BuildTree(filter(x -> x[:][node_min[1]] > node_min[2],S), node, Nodes, Position + 1, "False")
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

function TrainTest(S, test_size = 0.25)
    train = S[trunc(Int, floor( size(S,1) * test_size ) ) + 1 : size(S,1),:]
    test  = S[1:trunc(Int, floor( size(S,1) * test_size ) ),:]
    return train, test
end

function Accuracy(predictions)
    perc_right_answer = size(filter(x -> x[:][3] == true, predictions), 1) / size(predictions,1)
    return perc_right_answer * 100
end

function Predict(Test, predictions, nodes)
    Size_test   = size(Test,1)
    
    for row in eachrow(Test)
        node = GetRoot(nodes)
        while !node.isLeaf
            if row[node.feature] <= node.mean
                # True
                node = node.nodeTrue
            elseif row[node.feature] > node.mean
                # False
                node = node.nodeFalse
            end
        end
        if node.isLeaf
            pred_value  = row[size(Test,2)]
            result_node = (pred_value, unique(node.data[:,size(Test,2)])[1], unique(node.data[:,size(Test,2)])[1] == pred_value)
            push!(predictions, result_node)
        end
    end
end

function GetRoot(nodes)
    for node in nodes
        if node.way == "Root"
            return node
        end
    end
end

function TrainTree(train, nodes)
    # Create a pseudo root node
    # node_root   = Node(train, 0, 0.,false, "None", 0., 0)
    
    # Start creating the tree
    BuildTree(train, Any, nodes, 0)
end