import random
import time

def output(inputs,weight):
    y = 0
    for x,w in zip(inputs, weight):
        y += x * w
    if y > 0.0:
        return 1.0
    else:
        return 0.0

def calc_error(batchError, l_rate, error, inputs,expected): 
    
    batchError[0] = batchError[0] + (l_rate * error)
    for i in range(len(batchError)):
        if i != 0:
            batchError[i] = batchError[i] + (l_rate * error * inputs[i])
   
                


def train_weights(matrix, weights, n_epoch, l_rate,batch):
    batchError=[0,0,0]
    for epoch in range(n_epoch):
        for i in range(len(matrix)):

            #Find the Output y
            y = output(matrix[i][:-1], weights)
            
            #Calculate error
            error = matrix[i][-1] - y
            #print("Ouput epoch: {0} [ {1} , {2} ] = {3}/{4}".format(epoch,matrix[i][1],matrix[i][2],y,matrix[i][3]))
            
            if error != 0:
                calc_error(batchError,l_rate,error,matrix[i],matrix[i][-1])

            if (i+1)%batch == 0:
                for i in range(len(weights)):
                    weights[i]=weights[i]+batchError[i]
                batchError=[0,0,0]
        print("Epoc weight: {0}".format(weights))
                
    #Return the Final Weights
    print("Final Weights: {0}".format(weights))
    return weights

def main():
    start = time.process_time()
    n_epoch = 10
    l_rate = 1
    batch = 2
    #2 Input Nodes - 1 Bias - 1 Output

                 #  Bias    i1      i2      y
    matrix = [  [1.00,  0.0,    -0.5,    0.0],
                [1.00,  1.0,    -0.1,    0.0],
                [1.00,  0.5,    0.3,    1.0],
                [1.00,  0.1,    -0.1,    1.0],
                [1.00,  2.0,    0.5,    1.0],
                [1.00,  1.5,    0.5,    1.0]    
                ]

    #weights = [ random.uniform(-1.0,1.0),   random.uniform(-1.0,1.0),   random.uniform(-1.0,1.0)]
    weights = [ 1.0, -1.0,   -1.0]
    #will initialize randomly later

    train_weights(matrix, weights, n_epoch, l_rate,batch)

    #Find the Output using the trained model
    inputs = [1.0,    1.5,    0.5]
    y = output(inputs, weights)
    print("The Output for {0} inputs is: {1}".format(inputs[1:], y))
    print("Time taken")
    print("------------------------------------------------")
    print(time.process_time() - start)

if __name__ == '__main__':
    main()


