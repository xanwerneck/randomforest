using Statistics
using CSV

function TrainTest(S, test_size = 0.25)
    train = S[trunc(Int, floor( size(S,1) * test_size ) ) + 1 : size(S,1),:]
    test  = S[1:trunc(Int, floor( size(S,1) * test_size ) ),:]
    return train, test
end

function entropy(S)
    y_values  = S[:,2]
    y_uniques = unique(y_values)

    entropy = 0.
    for y_unique in y_uniques
        p_y_unique = count(x -> x == y_unique, y_values) / size(S,1)
        entropy += p_y_unique * log(p_y_unique)
    end
    return -entropy
end

function split_dataframe(S, feature, divider)
    if (feature_types[feature] == String)
        s_left  = filter(x -> x[:][feature] == divider, S)
        s_right = filter(x -> x[:][feature] !=  divider, S) 
    end
    if (feature_types[feature] == Int64)
        s_left  = filter(x -> x[:][feature] == divider, S)
        s_right = filter(x -> x[:][feature] !=  divider, S) 
    end
    if (feature_types[feature] == Float64)
        s_left  = filter(x -> x[:][feature] <= divider, S)
        s_right = filter(x -> x[:][feature] >  divider, S) 
    end
    return s_left, s_right
end

function informationgain(S, feature)
    
    unique_values = unique(S[:,1])

    max_gain = (0.,0,1)
    for unique_value in unique_values
        if (feature_types[feature] == String)
            s_left  = filter(x -> x[:][1] == unique_value, S)
            s_right = filter(x -> x[:][1] != unique_value, S) 
        end
        if (feature_types[feature] == Int64)
            s_left  = filter(x -> x[:][1] == unique_value, S)
            s_right = filter(x -> x[:][1] != unique_value, S) 
        end
        if (feature_types[feature] == Float64)
            s_left  = filter(x -> x[:][1] <= unique_value, S)
            s_right = filter(x -> x[:][1] > unique_value, S)    
        end

        gain = entropy(S) - (
            ( size(s_left,1) / size(S,1) ) * entropy(s_left)
           +( size(s_right,1) / size(S,1) ) * entropy(s_right)
        )
        if (gain > max_gain[1])
            max_gain = (gain, feature, unique_value)
        end
    end
    return max_gain

end

function createnode(S, divider, feature, way, node_left, node_right, isLeaf)
    node = [
        S,
        divider,
        feature,
        way,
        node_left,
        node_right,
        isLeaf
    ]
    return node
end

function buildtree(S, way = "I")
    N,M = size(S)

    # If node is Leaf - dont need to be divided    
    if ( size( unique(S[:,M]) , 1) == 1 )
        node = createnode(S, 0, 0, way, 0, 0, true)
        push!(nodes, node)
        return size(nodes,1)
    end

    gains = []
    for i in range(1,length=M-1)
        gain = informationgain(S[:,[i,M]], i)
        push!( gains, gain ) 
    end
    
    max_gain = gains[1]
    for gain in gains
        if (gain[1] > max_gain[1])
            max_gain = gain
        end
    end
    
    gain, feature, divider = max_gain
    
    s_left, s_right = split_dataframe(S, feature, divider)
    
    node_left  = buildtree(s_left, "L")
    node_right = buildtree(s_right, "R")

    node = createnode(S, divider, feature, way, node_left, node_right, false)
    push!(nodes, node)

    return size(nodes,1)
   
end

function getbegginner()
    for node in nodes
        if (node[4] == "I")
            return node
        end
    end
end

function check_predict(feature, value_1, value_2)
    if (feature_types[feature] == String)
        return value_1 == value_2
    end
    if (feature_types[feature] == Int64)
        return value_1 == value_2
    end
    if (feature_types[feature] == Float64)
        return value_1 <= value_2
    end
end

function predicttree(Test, predictions)
    Row_size_test, Col_size_test   = size(Test)

    for row in eachrow(Test)
        node = getbegginner()
        #node = nodes[size(nodes,1)] # Last node inserted
        
        while node[7] == false
            if check_predict(node[3], row[node[3]], node[2])
                # True
                node = nodes[node[5]]
            else
                # False
                node = nodes[node[6]]
            end
            
        end
        if node[7] == true
            pred_value_test  = row[Col_size_test]
            pred_value_tree  = unique(node[1][:,Col_size_test])[1]
            pred_value_check = row[Col_size_test] == unique(node[1][:,Col_size_test])[1]
            
            result_node = (pred_value_test, pred_value_tree, pred_value_check)
            push!(predictions, result_node)
        end
        
    end
end

function traindt(S)
    buildtree(S)
end

nodes = []
predictions = Array{Tuple{Any,Any,Bool}}(undef,0)

S_dataset   = CSV.read(string("data/",ARGS[1]))
# Build 2 dataframe - train with 75% and test with 25%
println("---- Split the data ----")
train, test = TrainTest(S_dataset, 0.25)

# Construct the feature types
feature_types = []
for j in range(1, length=size(train, 2))
    push!(feature_types, eltype(train[j]))
end

# Train and build a single tree
# Input: train : dataset of train
println("---- Train the data ----")
traindt(train)

# Predict the test dataset
println("---- Predict the test data ----")
predicttree(test, predictions)

mape = []
for i in range(1, length=size(predictions, 1))
    error_pred = abs(predictions[i][1] - predictions[i][2])
    push!(mape, 100 * ( error_pred / predictions[i][1] ) )
end
accuracy    = 100 - mean(mape)
@show accuracy
