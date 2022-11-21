import numpy as np
def matmul(mat0, mat1) :
    a0 = mat0.shape[0]
    a1 = mat0.shape[1]
    b0 = mat1.shape[0]
    b1 = mat1.shape[1]
    mat2 = np.zeros((a0,b1))
    for i in range(a0):
        for j in range(b1):
            for k in range(a1):
                mat2[i,j] += mat0[i,k]*mat1[k,j]
    return mat2
    
    

class NeuralNetwork:
    def __init__(self, layer_sizes, layer_activations, epsilon = 0.1, lr=0.01):
        self.epsilon = epsilon
        self.layer_sizes = layer_sizes
        self.layer_activations = layer_activations
        self.initialize()
        self.lr = lr
        self.activation_functions = {
            "relu":self.relu,
            "sigmoid":self.sigmoid,
            "grelu":self.grelu,
            "gsigmoid":self.gsigmoid            
            
            }
        
    def initialize(self):
        self.W = [] #NN weights
        self.b = [] #NN betas
        for i in range(len(self.layer_sizes)-1):
            self.W.append(np.random.randn(self.layer_sizes[i+1],self.layer_sizes[i]  )*self.epsilon) #Randomly initializes weights for all layers 
            self.b.append(np.random.randn(self.layer_sizes[i+1],1)*self.epsilon) #Randomly initializes betas for all layers 

    def sigmoid(self,x):
        return 1/(1+np.exp(-x))
    
    
    
    def gsigmoid(self,da,x):
        temp= self.sigmoid(x)
        return da * temp * (1-temp)
    
    
    def relu(self,x):
        return np.maximum(0,x)
    
    def grelu(self,da,x):

        temp = np.array(da, copy=True)
        temp[x<=0] = 0
        return temp
    

    def propagate(self,x,is_pred=False):
        cacheW = []
        z = x.T
        a = self.activation_functions[self.layer_activations[0]](z)
        
        
        for i in range(len(self.W)):
            cacheW.append((a,z))

            z = matmul(self.W[i], a ) + self.b[i]
    
            a = self.activation_functions[self.layer_activations[i]](z)
            #print(a.shape)
            assert(a.shape == z.shape)
        cacheW.append((a,z))

        if not is_pred:       
            
            self.cache =  cacheW
        else:
            return cacheW[-1][0]

    
    def backpropagate(self,y,y_pred):
        gradsW = []
        gradsb= []
  
        da= -(np.divide(y,y_pred+self.epsilon) - np.divide(1-y,1-y_pred+self.epsilon))

       
        for i in range(len(self.W))[::-1]:
            
            a_prev =  self.cache[i][0]

            z = self.cache[i+1][1] #Need to get the z for the next layer
            w = self.W[i]
            b = self.b[i]
            dz = self.activation_functions["g"+self.layer_activations[i]](da,z)      
            dw = matmul(dz, a_prev.T)/a_prev.shape[1]      
            db = np.sum(dz, axis=1,keepdims=True)/a_prev.shape[1]
           
           
            da = matmul(w.T, dz)
            
            gradsW.append(dw)
            gradsb.append(db)
            
        self.gradsw = gradsW
        self.gradsb = gradsb
        
    def update(self):
        
       
        for i in range(len(self.gradsw)):
            self.W[i] -= self.lr*self.gradsw[-(i+1)] #It updates the weights of the first layer, second layer... the grads lists are inverted because of the way they were stored (last first)
            self.b[i] -= self.lr*self.gradsb[-(i+1)]
      
    def fit(self,x,y,epochs=1):
        for i in range(epochs):
            self.propagate(x)
            self.backpropagate(y,self.cache[-1][0])
            self.update()  
    def predict(self,x):
        a =self.propagate(x,is_pred=True)
        return a