"""Neural Networks"""
import tensorflow as tf
import numpy as np
import sklearn
from sklearn.metrics import confusion_matrix

# pylint: disable=redefined-builtin, invalid-name, too-few-public-methods
class ConvLayer:
    """Convolutional Layer
    Attributes:
        filter_shape: [filter_height, filter_width, in_channels, out_channels]
    """
    def __init__(self, filter_shape, stddev=5e-2, dtype=tf.float32):
        self.filter = tf.Variable(
            tf.truncated_normal(
                shape=filter_shape,
                stddev=stddev,
                dtype=dtype))
        self.bias = tf.Variable(tf.zeros((filter_shape[-1],), dtype=tf.float32))

        conv_out = tf.nn.conv2d(
            input=input,
            filter=self.filter,
            strides=[1, 1, 1, 1],
            padding='VALID')

        self.out = tf.nn.relu(tf.nn.bias_add(conv_out, self.bias))

class ConvPoolLayer:
    """Convolutional layer and Max pooling layer
    Attributes:
        filter_shape: [filter_height, filter_width, in_channels, out_channels]
        ksize: pooling window
    """
    def __init__(self, filter_shape, ksize, dtype=tf.float32):
        self.filter_shape = filter_shape
        self.ksize = ksize

        fan_in = np.prod(filter_shape[:-1])
        fan_out = (np.prod(filter_shape[:2] * filter_shape[3]) / np.prod(ksize))
        bound = np.sqrt(6. / (fan_in + fan_out))
        self.filter = tf.Variable(
            tf.random_uniform(
                shape=filter_shape,
                minval=-bound,
                maxval=bound,
                dtype=dtype))
        self.bias = tf.Variable(tf.zeros((filter_shape[-1],), dtype=tf.float32))

    def predict(self, input):
        """perform convlution and pooling operations on given inputs"""
        conv_out = tf.nn.conv2d(
            input=input,
            filter=self.filter,
            strides=[1, 1, 1, 1],
            padding='VALID')
        conv_layer = tf.nn.relu(tf.nn.bias_add(conv_out, self.bias))

        pool_layer = tf.nn.max_pool(
            value=conv_layer,
            ksize=self.ksize,
            strides=[1, 1, 1, 1],
            padding='VALID')

        return pool_layer

class Dense:
    """Fully connected layer"""
    def __init__(self, n_in, n_out, keep_prob=0.5, dtype=tf.float32):
        self.keep_prob = keep_prob

        bound = np.sqrt(6. / (n_in + n_out))
        self.W = tf.Variable(
            tf.random_uniform(
                shape=[n_in, n_out],
                minval=-bound,
                maxval=bound,
                dtype=dtype))
        self.b = tf.Variable(tf.zeros((n_out,)))

    def predict(self, input, train=True):
        """fully connection with dropout options"""
        if train:
            input = tf.nn.dropout(input, self.keep_prob)
        else:
            pass

        #return tf.nn.bias_add(tf.matmul(input, self.w), self.b)
        return tf.nn.xw_plus_b(input, self.W, self.b, name='dense_out')
    
class LSTM2:
    def __init__(self, **paras):
        self.paras = paras
        
        if 'n_units' not in self.paras:
            self.paras['n_units'] = 64
            
        self.paras['n_out'] = 2

        self._init_weights()
        self._build_graph()
        
        
    def _init_weights(self):
        self.hid_to_out = tf.Variable(
            tf.random_uniform(
                shape=(self.paras['n_units'], self.paras['n_out']),
                minval=-0.1,
                maxval=0.1,
                dtype=tf.float32))
        self.hid_to_out_b = tf.Variable(tf.zeros((self.paras['n_out'],)))    
        
    def _forward_prop(self, X, keep_prob):
        X_expanded = tf.nn.embedding_lookup(self.paras['word_embeddings'], X)

        cell = tf.nn.rnn_cell.LSTMCell(num_units=self.paras['n_units'])
        output, state = tf.nn.dynamic_rnn(cell, X_expanded, dtype=tf.float32)

        output = tf.transpose(output, (1, 0, 2))[-1]

        logits = tf.nn.xw_plus_b(output, self.hid_to_out, self.hid_to_out_b)
        logits = tf.nn.dropout(logits, keep_prob)

        return logits
    
    def _build_graph(self):
        self.X = tf.placeholder(dtype=tf.int32, shape=(None, self.paras['sent_len']))
        self.Y = tf.placeholder(dtype=tf.int32, shape=(None,))
        self.keep_prob = tf.placeholder(dtype=tf.float32, name='keep_prob') 
        
        self.logits = self._forward_prop(self.X, self.keep_prob)

        # training
        self.loss = self.loss_func(self.Y, self.logits)
        optimizer = tf.train.RMSPropOptimizer(self.paras['lr'])
        self.step = optimizer.minimize(self.loss)

        # validating
        self.outputs = tf.nn.softmax(self.logits)
        Y_pred = tf.argmax(self.logits, axis=1, output_type=tf.int32)
        self.accuracy = tf.reduce_mean(
            tf.cast(tf.equal(Y_pred, self.Y), tf.float32))
        self.tp = tf.reduce_sum(tf.cast(Y_pred*self.Y, 'float'), axis=0)
        self.fp = tf.reduce_sum(tf.cast((1-Y_pred)*self.Y, 'float'), axis=0)
        self.fn = tf.reduce_sum(tf.cast(Y_pred*(1-self.Y), 'float'), axis=0)
        self.p = self.tp/(self.tp+self.fp+0.00001)
        self.r = self.tp/(self.tp+self.fn+0.00001)
        self.f1 = 2*self.p*self.r/(self.p+self.r+0.0001)
#         f1 = tf.where(tf.is_nan(f1), tf.zeros_like(f1), f1)
        auc = [tf.metrics.auc(labels=self.Y, predictions=Y_pred)]
        self.auc = [i[1] for i in auc]
        self.auc_op = [i[1] for i in auc]
        tf.local_variables_initializer().run()

    def loss_func(self, labels, logits):
        return tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=logits, labels=labels))    
        
class LSTM:
    """text classification model
    TODO: randomly initialised word vectors
    """
    def __init__(self, **paras):
        self.paras = paras
        
        if 'n_units' not in self.paras:
            self.paras['n_units'] = 64

        self._build_graph()

    def forward(self, X_indices):
        """forward propagation graph"""

        X_embeddings = tf.nn.embedding_lookup(self.paras['word_embeddings'], X_indices)

        rnn_cell = tf.nn.rnn_cell.LSTMCell(num_units=self.paras['n_units'])
        #initial_state = rnn_cell.zero_state(self.paras['batch_size'], dtype=tf.float32)

        outputs, state = tf.nn.dynamic_rnn(
            cell=rnn_cell,
            inputs=X_embeddings,
            dtype=tf.float32)#,
            #initial_state=initial_state)

        return tf.transpose(outputs, (1, 0, 2))[-1]

    def loss_func(self, labels, logits):
        """softmax loss for exclusive labels"""

        return tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=logits, labels=labels))

    def optimize(self, loss):
        """graph for optimisation TODO - learning rate decay"""
        optimizer = tf.train.GradientDescentOptimizer(self.paras['lr'])
        return optimizer.minimize(loss)

    def _build_graph(self):
        """graph for the full model"""

        self.X = tf.placeholder(
            dtype=tf.int32,
            shape=(self.paras['batch_size'], self.paras['sent_len']),
            name='inputs')
        self.Y = tf.placeholder(
            dtype=tf.int32,
            shape=(self.paras['batch_size'],),
            name='targets')
        
        #logits = self.forward(self.X)
        #logits_train = tf.nn.dropout(logits, self.paras['keep_prob'])
        rnn_out = self.forward(self.X)
        
        out_layer = Dense(                                                  
            keep_prob=self.paras['keep_prob'],                              
            n_in=self.paras['n_units'],                                                       
            n_out=2)
        
        #logits_train = out_layer.predict(rnn_out)
        #logits = out_layer.predict(rnn_out, train=False)
        logits = out_layer.predict(rnn_out)


        self.loss = self.loss_func(self.Y, logits)
        self.step = self.optimize(self.loss)

        output = tf.nn.softmax(logits)
        Y_pred = tf.argmax(output, axis=1, output_type=tf.int32)
        self.accuracy = tf.reduce_mean(
            tf.cast(tf.equal(Y_pred, self.Y), tf.float32))
        self.tp = tf.reduce_sum(tf.cast(Y_pred*self.Y, 'float'), axis=0)
        self.fp = tf.reduce_sum(tf.cast((1-Y_pred)*self.Y, 'float'), axis=0)
        self.fn = tf.reduce_sum(tf.cast(Y_pred*(1-self.Y), 'float'), axis=0)
        self.p = self.tp/(self.tp+self.fp+0.00001)
        self.r = self.tp/(self.tp+self.fn+0.00001)
        self.f1 = 2*self.p*self.r/(self.p+self.r+0.0001)
#         f1 = tf.where(tf.is_nan(f1), tf.zeros_like(f1), f1)
        auc = [tf.metrics.auc(labels=self.Y, predictions=Y_pred)]
        self.auc = [i[1] for i in auc]
        self.auc_op = [i[1] for i in auc]
        tf.global_variables_initializer().run()                                 
        tf.local_variables_initializer().run()

class STCNN:
    """text classification model
    TODO: randomly initialised word vectors
    """
    def __init__(self, **paras):
        self.paras = paras

        self._build_graph()

    def forward(self, X_indices):
        """forward propagation graph"""

        X_embeddings = tf.nn.embedding_lookup(self.paras['word_embeddings'], X_indices)
        X_4d = tf.expand_dims(X_embeddings, 3)

        conv_layers = []
        conv_outs = []
        for h in self.paras['filter_heights']:
            _conv_layer = ConvPoolLayer(
                filter_shape=(h, self.paras['embed_dim'], 1, self.paras['num_filter']),
                ksize=(1, self.paras['sent_len']-h+1, 1, 1))
            conv_layers.append(_conv_layer)

            _conv_out = _conv_layer.predict(X_4d)
            _conv_out_reshape = (_conv_out.shape[0], _conv_out.shape[-1])
            _conv_out = tf.reshape(_conv_out, _conv_out_reshape)
            conv_outs.append(_conv_out)
        conv_outs_flatten = tf.concat(conv_outs, axis=1)

        dense_layer = Dense(
            keep_prob=self.paras['keep_prob'],
            n_in=self.paras['num_filter']*len(self.paras['filter_heights']),
            n_out=2)

        logits = dense_layer.predict(conv_outs_flatten)
        return logits

    def loss_func(self, labels, logits):
        """softmax loss for exclusive labels"""

        return tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=logits, labels=labels))

    def optimize(self, loss):
        """graph for optimisation TODO - learning rate decay"""
        optimizer = tf.train.GradientDescentOptimizer(self.paras['lr'])
        return optimizer.minimize(loss)

    def _build_graph(self):
        """graph for the full model"""

        self.X = tf.placeholder(
            dtype=tf.int32,
            shape=(self.paras['batch_size'], self.paras['sent_len']),
            name='inputs')
        self.Y = tf.placeholder(
            dtype=tf.int32,
            shape=(self.paras['batch_size'],),
            name='targets')
        logits = self.forward(self.X)

        self.loss = self.loss_func(self.Y, logits)
        self.step = self.optimize(self.loss)
        output = tf.nn.softmax(logits)
        Y_pred = tf.argmax(output, axis=1, output_type=tf.int32)
        self.accuracy = tf.reduce_mean(
            tf.cast(tf.equal(Y_pred, self.Y), tf.float32))
        self.tp = tf.reduce_sum(tf.cast(Y_pred*self.Y, 'float'), axis=0)
        self.fp = tf.reduce_sum(tf.cast((1-Y_pred)*self.Y, 'float'), axis=0)
        self.fn = tf.reduce_sum(tf.cast(Y_pred*(1-self.Y), 'float'), axis=0)
        self.p = self.tp/(self.tp+self.fp+0.00001)
        self.r = self.tp/(self.tp+self.fn+0.00001)
        self.f1 = 2*self.p*self.r/(self.p+self.r+0.0001)
#         f1 = tf.where(tf.is_nan(f1), tf.zeros_like(f1), f1)
        auc = [tf.metrics.auc(labels=self.Y, predictions=Y_pred)]
        self.auc = [i[1] for i in auc]
        self.auc_op = [i[1] for i in auc]
        tf.global_variables_initializer().run()                                 
        tf.local_variables_initializer().run()

class STCNN2:
    """text classification model
    TODO: randomly initialised word vectors
    """
    def __init__(self, **paras):
        self.paras = paras
        self.paras['n_hidden'] = 100 

        self._init_weights()
        self._build_graph()
        
    def _init_weights(self):                                                    
        # shared hidden layer                                                   
        n_in_hidden = self.paras['num_filter']*len(self.paras['filter_heights'])
        n_out_hidden = self.paras['n_hidden']                                   
        bound = np.sqrt(6. / (n_in_hidden + n_out_hidden))                      
        self.convpool_to_hidden = tf.Variable(                                  
            tf.random_uniform(                                                  
                shape=[n_in_hidden, n_out_hidden],                           
                minval=-bound,                                                  
                maxval=bound,                                                   
                dtype=tf.float32))                                              
        self.b_convpool_to_hidden = tf.Variable(tf.zeros((n_out_hidden,)))    
                                                                                
        # task-specific hidden layer                                            
        n_in_output = n_out_hidden                                              
        n_out_output = 2                                                        
        bound = np.sqrt(6. / (n_in_output + n_out_output))                      
        self.hidden_to_output = tf.Variable(                                    
            tf.random_uniform(                                                  
                shape=[n_in_output, n_out_output],                           
                minval=-bound,                                                  
                maxval=bound,                                                   
                dtype=tf.float32))                                              
        self.b_hidden_to_output = tf.Variable(tf.zeros((n_out_output,)))

    def forward(self, X_indices, keep_prob):
        """forward propagation graph"""

        X_embeddings = tf.nn.embedding_lookup(self.paras['word_embeddings'], X_indices)
        X_4d = tf.expand_dims(X_embeddings, 3)

        conv_layers = []
        conv_outs = []
        for h in self.paras['filter_heights']:
            _conv_layer = ConvPoolLayer(
                filter_shape=(h, self.paras['embed_dim'], 1, self.paras['num_filter']),
                ksize=(1, self.paras['sent_len']-h+1, 1, 1))
            conv_layers.append(_conv_layer)

            _conv_out = _conv_layer.predict(X_4d)
            _conv_out_reshape = (_conv_out.shape[0], _conv_out.shape[-1])
            _conv_out = tf.reshape(_conv_out, _conv_out_reshape)
            conv_outs.append(_conv_out)
        conv_outs_flatten = tf.concat(conv_outs, axis=1)

        hidden_layer = tf.nn.xw_plus_b(                                     
                conv_outs_flatten,                                              
                self.convpool_to_hidden,                                  
                self.b_convpool_to_hidden,                                
                name='convpool_to_hidden')                                      
        hidden_layer = tf.nn.dropout(hidden_layer, keep_prob)
        out_layer = tf.nn.xw_plus_b(
                hidden_layer,
                self.hidden_to_output,
                self.b_hidden_to_output,
                name='hidden_to_output')
        out_layer = tf.nn.dropout(out_layer, keep_prob)
        
        return out_layer

    def loss_func(self, labels, logits):
        """softmax loss for exclusive labels"""

        return tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=logits, labels=labels))

    def optimize(self, loss):
        """graph for optimisation TODO - learning rate decay"""
        optimizer = tf.train.GradientDescentOptimizer(self.paras['lr'])
        return optimizer.minimize(loss)

    def _build_graph(self):
        """graph for the full model"""

        self.X = tf.placeholder(
            dtype=tf.int32,
            shape=(self.paras['batch_size'], self.paras['sent_len']),
            name='inputs')
        self.Y = tf.placeholder(
            dtype=tf.int32,
            shape=(self.paras['batch_size'],),
            name='targets')
        self.keep_prob = tf.placeholder(                                        
            dtype=tf.float32,                                                   
            name='keep_prob') 
        logits = self.forward(self.X, self.keep_prob)

        self.loss = self.loss_func(self.Y, logits)
        self.step = self.optimize(self.loss)
        output = tf.nn.softmax(logits)
        Y_pred = tf.argmax(output, axis=1, output_type=tf.int32)
        self.accuracy = tf.reduce_mean(
            tf.cast(tf.equal(Y_pred, self.Y), tf.float32))
        self.tp = tf.reduce_sum(tf.cast(Y_pred*self.Y, 'float'), axis=0)
        self.fp = tf.reduce_sum(tf.cast((1-Y_pred)*self.Y, 'float'), axis=0)
        self.fn = tf.reduce_sum(tf.cast(Y_pred*(1-self.Y), 'float'), axis=0)
        self.p = self.tp/(self.tp+self.fp+0.00001)
        self.r = self.tp/(self.tp+self.fn+0.00001)
        self.f1 = 2*self.p*self.r/(self.p+self.r+0.0001)
#         f1 = tf.where(tf.is_nan(f1), tf.zeros_like(f1), f1)
        auc = [tf.metrics.auc(labels=self.Y, predictions=Y_pred)]
        self.auc = [i[1] for i in auc]
        self.auc_op = [i[1] for i in auc]
        tf.global_variables_initializer().run()                                 
        tf.local_variables_initializer().run()

class MTCNN:                                                                    
    """text classification model"""                                                                         
    def __init__(self, **paras):                                                
        self.paras = paras                                                      
                                                                                
        self._build_graph()                                                     
                                                                                
    def forward(self, X_indices):                                               
        """forward propagation graph"""                                         
                                                                                
        X_embeddings = tf.nn.embedding_lookup(self.paras['word_embeddings'], X_indices)
        X_4d = tf.expand_dims(X_embeddings, 3)                                  
                                                                                
        conv_layers = []                                                        
        conv_outs = []                                                          
        for h in self.paras['filter_heights']:                                  
            _conv_layer = ConvPoolLayer(                                        
                filter_shape=(h, self.paras['embed_dim'], 1, self.paras['num_filter']),
                ksize=(1, self.paras['sent_len']-h+1, 1, 1))                    
            conv_layers.append(_conv_layer)                                     
                                                                                
            _conv_out = _conv_layer.predict(X_4d)                               
            _conv_out_reshape = (_conv_out.shape[0], _conv_out.shape[-1])       
            _conv_out = tf.reshape(_conv_out, _conv_out_reshape)                
            conv_outs.append(_conv_out)                                         
        conv_outs_flatten = tf.concat(conv_outs, axis=1)                        
                                                                                
        out_layers = []                                                         
        for task in range(self.paras['n_tasks']):                               
            hidden_layer = Dense(                                               
                keep_prob=self.paras['keep_prob'],                              
                n_in=self.paras['num_filter']*len(self.paras['filter_heights']),
                n_out=100)                                                      
            _logits = hidden_layer.predict(conv_outs_flatten)                   
            out_layer = Dense(                                                  
                keep_prob=self.paras['keep_prob'],                              
                n_in=100,                                                       
                n_out=2)                                                        
            _logits_out = out_layer.predict(_logits)                            
            out_layers.append(_logits_out)                                      
        #logits = tf.concat(out_layers, axis=1)                                 
                                                                                
        return out_layers                                                       
    def loss_func(self, labels, logits):                                        
        return tf.reduce_sum([self._single_task_cost_func(labels[:, i], logits[i]) for i in range(labels.shape[1])])
                                                                                
    def _single_task_cost_func(self, labels, logits):                           
        """softmax loss for exclusive labels"""                                 
                                                                                
        return tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(   
            logits=logits, labels=labels))                                      
                                                                                
    def optimize(self, loss):                                                   
        """graph for optimisation TODO - learning rate decay"""                 
        optimizer = tf.train.GradientDescentOptimizer(self.paras['lr'])         
        return optimizer.minimize(loss)                                         
                                                                                
    def _build_graph(self):                                                     
        """graph for the full model"""                                          
                                                                                
        self.X = tf.placeholder(                                                
            dtype=tf.int32,                                                     
            shape=(self.paras['batch_size'], self.paras['sent_len']),           
            name='inputs')                                                      
        self.Y = tf.placeholder(                                                
            dtype=tf.int32,                                                     
            shape=(self.paras['batch_size'], self.paras['n_tasks']),            
            name='targets')                                                     
        logits = self.forward(self.X)                                           
                                                                                
        self.loss = self.loss_func(self.Y, logits)                              
        self.step = self.optimize(self.loss)                                    
                                                                                
        outputs = [tf.nn.softmax(logits[i]) for i in range(self.paras['n_tasks'])]
        Y_pred = [tf.argmax(outputs[i], axis=1, output_type=tf.int32) for i in range(self.paras['n_tasks'])]
        self.accuracy = [tf.reduce_mean(tf.cast(tf.equal(Y_pred[i], self.Y[:, i]), tf.float32)) for i in range(self.paras['n_tasks'])]
        
        self.tp = [tf.reduce_sum(tf.cast(Y_pred[i]*self.Y[:, i], 'float'), axis=0) for i in range(self.paras['n_tasks'])]
        self.fp = [tf.reduce_sum(tf.cast((1-Y_pred[i])*self.Y[:, i], 'float'), axis=0) for i in range(self.paras['n_tasks'])]
        self.fn = [tf.reduce_sum(tf.cast(Y_pred[i]*(1-self.Y[:, i]), 'float'), axis=0) for i in range(self.paras['n_tasks'])]
        self.p = [self.tp[i]/(self.tp[i]+self.fp[i]+0.00001) for i in range(self.paras['n_tasks'])]
        self.r = [self.tp[i]/(self.tp[i]+self.fn[i]+0.00001) for i in range(self.paras['n_tasks'])]
        self.f1 = [2*self.p[i]*self.r/(self.p[i]+self.r[i]+0.0001) for i in range(self.paras['n_tasks'])]
        auc = [tf.metrics.auc(labels=self.Y[:, i], predictions=Y_pred[i]) for i in range(self.paras['n_tasks'])]
        self.auc = [i[1] for i in auc]
        self.auc_op = [i[1] for i in auc]
        


#         self.fscore = [tf.contrib.metrics.f1_score(Y_pred[i], self.Y[:, i]) for i in range(self.paras['n_tasks'])]
#         print (self.fscore[0])
                                                                                
        tf.global_variables_initializer().run()                                 
        tf.local_variables_initializer().run()
        
class MTCNN2:                                                                  
    """text classification model"""                                             
    def __init__(self, **paras):                                                
        self.paras = paras                                                      
        self.paras['n_hidden'] = 100                                            
                                                                                
        self._init_weights()                                                    
        self._build_graph()                                                     
                                                                                
    def _init_weights(self):                                                    
        # shared hidden layer                                                   
        n_in_hidden = self.paras['num_filter']*len(self.paras['filter_heights'])
        n_out_hidden = self.paras['n_hidden']                                   
        bound = np.sqrt(6. / (n_in_hidden + n_out_hidden))                      
        self.convpool_to_hidden = tf.Variable(                                  
            tf.random_uniform(                                                  
                shape=[3, n_in_hidden, n_out_hidden],                           
                minval=-bound,                                                  
                maxval=bound,                                                   
                dtype=tf.float32))                                              
        self.b_convpool_to_hidden = tf.Variable(tf.zeros((3, n_out_hidden)))    
                                                                                
        # task-specific hidden layer                                            
        n_in_output = n_out_hidden                                              
        n_out_output = 2                                                        
        bound = np.sqrt(6. / (n_in_output + n_out_output))                      
        self.hidden_to_output = tf.Variable(                                    
            tf.random_uniform(                                                  
                shape=[3, n_in_output, n_out_output],                           
                minval=-bound,                                                  
                maxval=bound,                                                   
                dtype=tf.float32))                                              
        self.b_hidden_to_output = tf.Variable(tf.zeros((3, n_out_output)))      
                                                                                
    def forward(self, X_indices, keep_prob):                                    
        """forward propagation graph"""                                         
                                                                                
        X_embeddings = tf.nn.embedding_lookup(self.paras['word_embeddings'], X_indices)
        X_4d = tf.expand_dims(X_embeddings, 3)                                  
                                                                                
        conv_layers = []                                                        
        conv_outs = []                                                          
        for h in self.paras['filter_heights']:                                  
            _conv_layer = ConvPoolLayer(                                        
                filter_shape=(h, self.paras['embed_dim'], 1, self.paras['num_filter']),
                ksize=(1, self.paras['sent_len']-h+1, 1, 1))                    
            conv_layers.append(_conv_layer)                                     
                                                                                
            _conv_out = _conv_layer.predict(X_4d)                               
            _conv_out_reshape = (_conv_out.shape[0], _conv_out.shape[-1])       
            _conv_out = tf.reshape(_conv_out, _conv_out_reshape)                
            conv_outs.append(_conv_out)                                         
        conv_outs_flatten = tf.concat(conv_outs, axis=1)                        
                                                                                
        out_layers = []                                                         
        for task in range(self.paras['n_tasks']):           
            hidden_layer = tf.nn.xw_plus_b(                                     
                conv_outs_flatten,                                              
                self.convpool_to_hidden[task],                                  
                self.b_convpool_to_hidden[task],                                
                name='convpool_to_hidden')                                      
            hidden_layer = tf.nn.dropout(hidden_layer, keep_prob)
            out_layer = tf.nn.xw_plus_b(
                hidden_layer,
                self.hidden_to_output[task],
                self.b_hidden_to_output[task],
                name='hidden_to_output')
            out_layer = tf.nn.dropout(out_layer, keep_prob)
            out_layers.append(out_layer)

        return out_layers
                                                                                
    def loss_func(self, labels, logits):                                        
        return tf.reduce_sum([self._single_task_cost_func(labels[:, i], logits[i]) for i in range(labels.shape[1])])
                                                                                
    def _single_task_cost_func(self, labels, logits):                           
        """softmax loss for exclusive labels"""                                 
                                                                                
        return tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(   
            logits=logits, labels=labels))                                      
                                                                                
    def optimize(self, loss):                                                   
        """graph for optimisation TODO - learning rate decay"""
        
        if self.paras['opt'] == 'sgd':                 
            optimizer = tf.train.GradientDescentOptimizer(self.paras['lr'])
        elif self.paras['opt'] == 'momentum':
            optimizer = tf.train.MomentumOptimizer(self.paras['lr'], 0.9)
        elif self.paras['opt'] == 'nesterov_momentum':
            optimizer = tf.train.MomentumOptimizer(self.paras['lr'], 0.9, use_nesterov=True)
        elif self.paras['opt'] == 'adagrad':
            optimizer = tf.train.AdagradOptimizer(self.paras['lr'])            
        elif self.paras['opt'] == 'rmsprop':
            optimizer = tf.train.RMSPropOptimizer(self.paras['lr'])
        elif self.paras['opt'] == 'adadelta':
            optimizer = tf.train.AdadeltaOptimizer(self.paras['lr'])
        elif self.paras['opt'] == 'adam':
            optimizer = tf.train.AdamOptimizer(self.paras['lr'])
        return optimizer.minimize(loss)                                         
                                                                                
    def _build_graph(self):                                                     
        """graph for the full model"""                                          
                                                                                
        self.X = tf.placeholder(                                                
            dtype=tf.int32,                                                     
            shape=(self.paras['batch_size'], self.paras['sent_len']),           
            name='inputs')                                                      
        self.Y = tf.placeholder(                                                
            dtype=tf.int32,                                                     
            shape=(self.paras['batch_size'], self.paras['n_tasks']),            
            name='targets')                                                     
        self.keep_prob = tf.placeholder(                                        
            dtype=tf.float32,                                                   
            name='keep_prob')                                                   
                                                                                
        logits = self.forward(self.X, self.keep_prob)                           
                                                                                
        self.loss = self.loss_func(self.Y, logits)
        self.loss_all = [self._single_task_cost_func(self.Y[:, i], logits[i]) for i in range(self.Y.shape[1])]                              
        self.step = self.optimize(self.loss)                                    
                                                                                
        outputs = [tf.nn.softmax(logits[i]) for i in range(self.paras['n_tasks'])]
        Y_pred = [tf.argmax(outputs[i], axis=1, output_type=tf.int32) for i in range(self.paras['n_tasks'])]
        self.accuracy = [tf.reduce_mean(tf.cast(tf.equal(Y_pred[i], self.Y[:, i]), tf.float32)) for i in range(self.paras['n_tasks'])]

        self.tp = [tf.reduce_sum(tf.cast(Y_pred[i]*self.Y[:, i], 'float'), axis=0) for i in range(self.paras['n_tasks'])]
        self.fp = [tf.reduce_sum(tf.cast((1-Y_pred[i])*self.Y[:, i], 'float'), axis=0) for i in range(self.paras['n_tasks'])]
        self.fn = [tf.reduce_sum(tf.cast(Y_pred[i]*(1-self.Y[:, i]), 'float'), axis=0) for i in range(self.paras['n_tasks'])]
        self.p = [self.tp[i]/(self.tp[i]+self.fp[i]+0.00001) for i in range(self.paras['n_tasks'])]
        self.r = [self.tp[i]/(self.tp[i]+self.fn[i]+0.00001) for i in range(self.paras['n_tasks'])]
        self.f1 = [2*self.p[i]*self.r/(self.p[i]+self.r[i]+0.0001) for i in range(self.paras['n_tasks'])]
        auc = [tf.metrics.auc(labels=self.Y[:, i], predictions=Y_pred[i]) for i in range(self.paras['n_tasks'])]
        self.auc = [i[1] for i in auc]
        self.auc_op = [i[1] for i in auc]        
