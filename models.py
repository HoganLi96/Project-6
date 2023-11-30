import nn
 
class PerceptronModel(object):
    def __init__(self, dimensions):
        self.w = nn.Parameter(1, dimensions)
 
    def get_weights(self):
        return self.w
 
    def run(self, x):
        node = nn.DotProduct(x, self.w)
        return node
 
    def get_prediction(self, x):
        y=self.run(x)
        if nn.as_scalar(y) >= 0.0:
            return 1
        else:
            return -1
    # def train(self, dataset):
        
    #     f=1
    #     while f==1:
    #         f=0
    #         for x, y in dataset.iterate_once(1):
    #             if self.get_prediction(x) != nn.as_scalar(y):
    #                 nn.Parameter.update(self.w,x,nn.as_scalar(y))
    #                 f=1    
    def train(self, dataset):
        while True:
            success = True
            for x, y in dataset.iterate_once(1):
                if self.get_prediction(x) != nn.as_scalar(y): 
                    success = False
                    nn.Parameter.update(self.w, x, nn.as_scalar(y))
            if success:
                break
 
 
class RegressionModel(object):
    def __init__(self):
        self.batch_size = 1
        self.w0 = nn.Parameter(1, 50)
        self.b0 = nn.Parameter(1, 50)
        self.w1 = nn.Parameter(50, 1)
        self.b1 = nn.Parameter(1, 1)

    # def run(self, x):
    #     xw1 = nn.Linear(x, self.w0)
    #     r1 = nn.ReLU(nn.AddBias(xw1, self.b0))
    #     xw2 = nn.Linear(r1, self.w1)
    #     return nn.AddBias(xw2, self.b1)
    # def run(self, x):

    #     fx1 = nn.AddBias(nn.Linear(x, self.w0), self.b0)
    #     relu1 = nn.ReLU(fx1)
    #     fx2 = nn.AddBias(nn.Linear(relu1, self.w1), self.b1)
    #     return fx2
    def run(self, x):

        xw1 = nn.AddBias(nn.Linear(x, self.w0), self.b0)
        relu1 = nn.ReLU(xw1)
        xw2 = nn.AddBias(nn.Linear(relu1, self.w1), self.b1)
        return xw2

    def get_loss(self, x, y):

 
        return nn.SquareLoss(self.run(x), y)
 
    def train(self, dataset):
        """
        Trains the model.
        """
        "*** YOUR CODE HERE ***"
        while True:
 
            
 
            for x, y in dataset.iterate_once(self.batch_size):
                loss = self.get_loss(x, y)
                grad = nn.gradients(loss, [self.w0, self.w1, self.b0, self.b1])
 
                
                self.w0.update(grad[0], -0.005)
                self.w1.update(grad[1], -0.005)
                self.b0.update(grad[2], -0.005)
                self.b1.update(grad[3], -0.005)

            print(nn.as_scalar(self.get_loss(nn.Constant(dataset.x), nn.Constant(dataset.y))))
 
            if nn.as_scalar(self.get_loss(nn.Constant(dataset.x), nn.Constant(dataset.y))) < 0.02:
                return
 
 
class DigitClassificationModel(object):

    # def __init__(self):
    #     # Initialize your model parameters here
    #     "*** YOUR CODE HERE ***"
    #     self.batch_size = 1
    #     self.w0 = nn.Parameter(784, 100)
    #     self.b0 = nn.Parameter(1, 100)
    #     self.w1 = nn.Parameter(100, 10)
    #     self.b1 = nn.Parameter(1, 10)
    def __init__(self):
        # Initialize your model parameters here
        "*** YOUR CODE HERE ***"
        self.batch_size = 1
        self.w0 = nn.Parameter(784, 250)
        self.b0 = nn.Parameter(1, 250)
        self.w1 = nn.Parameter(250, 150)
        self.b1 = nn.Parameter(1, 150)
        self.w2 = nn.Parameter(150,10)
        self.b2 = nn.Parameter(1, 10)

    # def run(self, x):
    #     fx1 = nn.AddBias(nn.Linear(x, self.w0), self.b0)
    #     relu1 = nn.ReLU(fx1)
    #     fx2 = nn.AddBias(nn.Linear(relu1, self.w1), self.b1)
    #     return fx2
    def run(self, x):
        fx1 = nn.AddBias(nn.Linear(x, self.w0), self.b0)
        relu1 = nn.ReLU(fx1)
        fx2 = nn.AddBias(nn.Linear(relu1, self.w1), self.b1)
        relu2 = nn.ReLU(fx2)
        fx3 = nn.AddBias(nn.Linear(relu2, self.w2), self.b2)
        return fx3
    
    def get_loss(self, x, y):
        """
        Computes the loss for a batch of examples.
 
        The correct labels `y` are represented as a node with shape
        (batch_size x 10). Each row is a one-hot vector encoding the correct
        digit class (0-9).
 
        Inputs:
            x: a node with shape (batch_size x 784)
            y: a node with shape (batch_size x 10)
        Returns: a loss node
        """
        "*** YOUR CODE HERE ***"
 
        return nn.SoftmaxLoss(self.run(x), y)
 
 
    def train(self, dataset):
        """
        Trains the model.
        """
        "*** YOUR CODE HERE ***"
        while True:
 
 
            for x, y in dataset.iterate_once(self.batch_size):
                loss = self.get_loss(x, y)
                grad = nn.gradients(loss, [self.w0, self.w1, self.w2, self.b0, self.b1, self.b2])
 
                self.w0.update(grad[0], -0.005)
                self.w1.update(grad[1], -0.005)
                self.w2.update(grad[2], -0.005)
                self.b0.update(grad[3], -0.005)
                self.b1.update(grad[4], -0.005)
                self.b2.update(grad[5], -0.005)

            print(dataset.get_validation_accuracy())
            if dataset.get_validation_accuracy() >= 0.97:
                return
 
 
class LanguageIDModel(object):
    """
    A model for language identification at a single-word granularity.
 
    (See RegressionModel for more information about the APIs of different
    methods here. We recommend that you implement the RegressionModel before
    working on this part of the project.)
    """
    def __init__(self):
        # Our dataset contains words from five different languages, and the
        # combined alphabets of the five languages contain a total of 47 unique
        # characters.
        # You can refer to self.num_chars or len(self.languages) in your code
        self.num_chars = 47
        self.languages = ["English", "Spanish", "Finnish", "Dutch", "Polish"]
 
        # Initialize your model parameters here
        "*** YOUR CODE HERE ***"
 
    def run(self, xs):
        """
        Runs the model for a batch of examples.
 
        Although words have different lengths, our data processing guarantees
        that within a single batch, all words will be of the same length (L).
 
        Here `xs` will be a list of length L. Each element of `xs` will be a
        node with shape (batch_size x self.num_chars), where every row in the
        array is a one-hot vector encoding of a character. For example, if we
        have a batch of 8 three-letter words where the last word is "cat", then
        xs[1] will be a node that contains a 1 at position (7, 0). Here the
        index 7 reflects the fact that "cat" is the last word in the batch, and
        the index 0 reflects the fact that the letter "a" is the inital (0th)
        letter of our combined alphabet for this task.
 
        Your model should use a Recurrent Neural Network to summarize the list
        `xs` into a single node of shape (batch_size x hidden_size), for your
        choice of hidden_size. It should then calculate a node of shape
        (batch_size x 5) containing scores, where higher scores correspond to
        greater probability of the word originating from a particular language.
 
        Inputs:
            xs: a list with L elements (one per character), where each element
                is a node with shape (batch_size x self.num_chars)
        Returns:
            A node with shape (batch_size x 5) containing predicted scores
                (also called logits)
        """
        "*** YOUR CODE HERE ***"
 
    def get_loss(self, xs, y):
        """
        Computes the loss for a batch of examples.
 
        The correct labels `y` are represented as a node with shape
        (batch_size x 5). Each row is a one-hot vector encoding the correct
        language.
 
        Inputs:
            xs: a list with L elements (one per character), where each element
                is a node with shape (batch_size x self.num_chars)
            y: a node with shape (batch_size x 5)
        Returns: a loss node
        """
        "*** YOUR CODE HERE ***"
 
    def train(self, dataset):
        """
        Trains the model.
        """
        "*** YOUR CODE HERE ***"