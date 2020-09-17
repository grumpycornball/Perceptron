import random

def output(inputs,weight):
    y = 0
    for x,w in zip(inputs, weight):
        y += x * w
    if y > 0.0:
        return 1.0
    else:
        return 0.0

def update_weights(weights, l_rate, error, inputs,expected): 
    
    weights[0] = weights[0] + (l_rate * error)
    for i in range(len(weights)):
        if i != 0:
            weights[i] = weights[i] + (l_rate * error * inputs[i])    
   
                


def train_weights(matrix, weights, n_epoch, l_rate):
    for epoch in range(n_epoch):        #looping through epocs
        for i in range(len(matrix)):

            #Find the Output y
            y = output(matrix[i][:-1], weights)
            
            #Calculate error
            error = matrix[i][-1] - y
            print("Ouput epoch: {0} [ {1} , {2} ] = {3}/{4}".format(epoch,matrix[i][1],matrix[i][2],y,matrix[i][3]))
            while error !=0:            
                #Updates the Weights in each epoch
                update_weights(weights, l_rate, error, matrix[i],matrix[i][-1])
                y = output(matrix[i][:-1], weights)
                error = matrix[i][-1] - y
                print("\t Adjusted weights"+ str(weights))
                print("Ouput epoch: {0} [ {1} , {2} ] = {3}/{4}".format(epoch,matrix[i][1],matrix[i][2],y,matrix[i][3]))

                
    #Return the Final Weights
    print("Final Weights: {0}".format(weights))
    return weights

def main():

    n_epoch = 10
    l_rate = 1

    #2 Input Nodes - 1 Bias - 1 Output

                 #  Bias    i1      i2      y
    matrix = [  [1.00,  0.0,    0.0,    1.0],
                [1.00,  0.0,    1.0,    1.0],
                [1.00,  1.0,    0.0,    1.0],
                [1.00,  1.0,    1.0,    0.0],]

    #weights = [ random.uniform(-1.0,1.0),   random.uniform(-1.0,1.0),   random.uniform(-1.0,1.0)]
    weights = [ 1.0, -1.0,   -1.0]
    #will initialize randomly later

    train_weights(matrix, weights, n_epoch, l_rate)

    #Find the Output using the trained model
    inputs = [1.0,    0.0,    0.0]
    y = output(inputs, weights)
    print("The Output for {0} inputs is: {1}".format(inputs[1:], y))
    inputs = [1.0,    0.0,    1.0]
    y = output(inputs, weights)
    print("The Output for {0} inputs is: {1}".format(inputs[1:], y))
    inputs = [1.0,    1.0,    0.0]
    y = output(inputs, weights)
    print("The Output for {0} inputs is: {1}".format(inputs[1:], y))
    inputs = [1.0,    1.0,    1.0]
    y = output(inputs, weights)
    print("The Output for {0} inputs is: {1}".format(inputs[1:], y))

if __name__ == '__main__':
    main()


