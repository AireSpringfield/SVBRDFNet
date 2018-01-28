import tensorflow as tf
import numpy as np
import math


class BaseNet(object):
   

    def Filter(self, shape):
        initializer=tf.contrib.layers.xavier_initializer()
        return tf.Variable(initializer(shape), name='filter')

    def Weights(self, shape):
        initializer=tf.contrib.layers.xavier_initializer()
        return tf.Variable(initializer(shape), name='weights')

    def Bias(self, shape):
        return tf.Variable(tf.zeros(shape), name='bias')

    def Conv(self, input, filter, strides=[1,1,1,1], padding = 'SAME', use_cudnn_on_gpu=True, data_format='NHWC'):
        return tf.nn.conv2d(input, filter, strides, padding, use_cudnn_on_gpu, data_format)

    '''
    def DeConv(self, value, filter, output_shape, strides=[1,1,1,1], padding='SAME', data_format='NHWC'):
        return tf.nn.conv2d_transpose(value, filter, output_shape, strides, padding, data_format)
    '''

    def BilinearUpsampleDoubleResolution(self, input):
        shape = input.shape
        new_size = [int(shape[1])*2, int(shape[2])*2]
        return tf.image.resize_bilinear(input, new_size)


    def BatchNorm(self, input, scope):
        # return tf.layers.batch_normalization(input, axis=-1, center=True, scale=True, beta_initializer=tf.constant_initializer(0.0001), gamma_initializer=tf.constant_initializer(1.00001), epsilon=0.0001, training=self.is_training, trainable=False, scope=scope+'/BN')
        return tf.contrib.layers.batch_norm(input, center=True, scale=True, epsilon=0.0001, trainable=True, scope=scope+'/BN')
  
    def ReLU(self, input):
        return tf.nn.relu(input)

    ###########################################

    def FC(self, input, out_channels, withReLU=True, name=None):
        with tf.name_scope(name):
            shape=input.get_shape().as_list() # tensor.get_shape() can be used only if the shape of a tensor can be deduced directly from the graph
            if len(shape)==4:
                in_channels=shape[1]*shape[2]*shape[3]
            else:
                in_channels=shape[1]
            weights=self.Weights([in_channels, out_channels])
            bias=self.Bias([out_channels])
            flatten=tf.reshape(input, [-1, in_channels])
            if withReLU:
                return self.ReLU(tf.matmul(flatten, weights)+bias)
            else:
                return tf.matmul(flatten, weights)+bias


    def ConvSameResolution(self,  input, filter_shape, strides=[1,1,1,1], padding = 'SAME', use_cudnn_on_gpu=True, data_format='NHWC', withBN=True, name=None):
        with tf.name_scope(name):
            conv_filter = self.Filter(filter_shape)
            conv_bias=self.Bias(filter_shape[3])
            conv=self.Conv(input, conv_filter, strides, padding, use_cudnn_on_gpu, data_format)+conv_bias
            if withBN:
                bn=self.BatchNorm(conv, name)
                return self.ReLU(bn)
            else:
                return self.ReLU(conv)
    
    def ConvSameResolutionSigmoid(self,  input, filter_shape, strides=[1,1,1,1], padding = 'SAME', use_cudnn_on_gpu=True, data_format='NHWC', withBN=False, name=None):
        with tf.name_scope(name):
            conv_filter = self.Filter(filter_shape)
            conv_bias=self.Bias(filter_shape[3])
            conv=self.Conv(input, conv_filter, strides, padding, use_cudnn_on_gpu, data_format)+conv_bias
            return tf.sigmoid(conv)
        

    def Concat(self, input_list, axis = 3, name=None): # default axis=3 to concat along "channels"
        return tf.concat(input_list, axis, name=name)

    def ConvHalfResolutionNoPooling(self,  input, filter_shape, strides=[1,2,2,1], padding = 'SAME', use_cudnn_on_gpu=True, data_format='NHWC', name=None):
            with tf.name_scope(name):
                conv_filter=self.Filter(filter_shape)
                conv_bias=self.Bias(filter_shape[3])
                conv=self.Conv(input, conv_filter, strides, padding, use_cudnn_on_gpu, data_format)+conv_bias
                bn=self.BatchNorm(conv, name)
                return self.ReLU(bn)
             


    def DeConvDoubleResolutionBilinear(self, input, filter_shape, strides=[1,1,1,1], padding='SAME', use_cudnn_on_gpu=True, data_format='NHWC', name=None):
        with tf.name_scope(name):
            conv=self.ConvSameResolution(input, filter_shape, strides, padding, use_cudnn_on_gpu, data_format, True, name)
            return self.BilinearUpsampleDoubleResolution(conv)

    def MSELoss(self, labels, predictions, name=None):
        with tf.name_scope(name):
            return tf.losses.mean_squared_error(labels, predictions)
    


class SVBRDFNet(BaseNet):

    params = {}
    params['nFilterFirstConv'] = 16


    def __init__(self, config):
        self.sess = tf.Session(config=config)
        self.is_training=tf.placeholder(tf.bool) # thisbool flag mainly affects BN layers
        self.total_iter=tf.Variable(0, trainable=False)
    
    def __del__(self):
        self.sess.close()
        

    def LoadData(self, batchSize=16, nInputChannels = 3, height = 256, width = 256):
        with tf.name_scope('Input'):
            self.data_image = tf.placeholder(tf.float32, [None, height, width, nInputChannels], name = 'data_image')
            self.data_albedo = tf.placeholder(tf.float32, [None, height, width, nInputChannels], name = 'data_albedo')
            self.data_spec = tf.placeholder(tf.float32, [None, nInputChannels], name = 'data_spec')
            self.data_roughness = tf.placeholder(tf.float32, [None, 1], name = 'data_roughness')
            self.data_normal = tf.placeholder(tf.float32, [None, height, width, nInputChannels], name = 'data_normal')

        # default data shape: NHWC
        

    def CreateNet(self, batchSize = 16, nInputChannels = 3,  nFilterFirstConv = 16, nFisrtFC=1024, weightMSE = [1.0, 1.0, 1.0, 1.0], kh=3, kw=3):

        

        lossweight_d = weightMSE[0]
        lossweight_s = weightMSE[1]
        lossweight_r = weightMSE[2]
        lossweight_n = weightMSE[3]

        self.LoadData(batchSize)
        
       
        

        # dictionary for tensorflow intermediate variables
        var={}

        
        #outchannalDict = {'Albedo':0, 'Spec':1, 'Roughness':2, 'Normal':3, 'Full':4, 'A-S':5}
        # conv layers
  
        for i in [0, 1, 2, 3]:                                                                                                                                                  # (height, width, nChannels)
            var['Conv0_ch{}'.format(i)]=self.ConvSameResolution(self.data_image, [kh, kw, nInputChannels, nFilterFirstConv], name='Conv0_ch{}'.format(i)) # 256*256*3 -> 256*256*16
            var['Conv1_ch{}'.format(i)]=self.ConvHalfResolutionNoPooling(var['Conv0_ch{}'.format(i)], [kh, kw, nFilterFirstConv, nFilterFirstConv*2], name='Conv1_ch{}'.format(i)) # 256*256*16 -> 128*128*32
            var['Conv2_ch{}'.format(i)]=self.ConvHalfResolutionNoPooling(var['Conv1_ch{}'.format(i)], [kh, kw, nFilterFirstConv*2, nFilterFirstConv*4], name='Conv2_ch{}'.format(i)) # 128*128*32 -> 64*64*64
            var['Conv3_ch{}'.format(i)]=self.ConvHalfResolutionNoPooling(var['Conv2_ch{}'.format(i)], [kh, kw, nFilterFirstConv*4, nFilterFirstConv*8], name='Conv3_ch{}'.format(i)) # 64*64*64 -> 32*32*128
            var['Conv4_ch{}'.format(i)]=self.ConvHalfResolutionNoPooling(var['Conv3_ch{}'.format(i)], [kh, kw, nFilterFirstConv*8, nFilterFirstConv*16], name='Conv4_ch{}'.format(i)) # 32*32*128 -> 16*16*256
            var['Conv5_ch{}'.format(i)]=self.ConvHalfResolutionNoPooling(var['Conv4_ch{}'.format(i)], [kh, kw, nFilterFirstConv*16, nFilterFirstConv*16], name='Conv5_ch{}'.format(i)) # 16*16*256 -> 8*8*256

            var['MidConv0_ch{}'.format(i)]=self.ConvSameResolution(var['Conv5_ch{}'.format(i)],[kh, kw, nFilterFirstConv*16, nFilterFirstConv*16], name='MidConv0_ch{}'.format(i)) # 8*8*256 -> 8*8*256
            var['MidConv1_ch{}'.format(i)]=self.ConvSameResolution(var['MidConv0_ch{}'.format(i)],[kh, kw, nFilterFirstConv*16, nFilterFirstConv*16], name='MidConv1_ch{}'.format(i)) # 8*8*256 -> 8*8*256
            var['MidConv2_ch{}'.format(i)]=self.ConvSameResolution(var['MidConv1_ch{}'.format(i)],[kh, kw, nFilterFirstConv*16, nFilterFirstConv*16], name='MidConv2_ch{}'.format(i)) # 8*8*256 -> 8*8*256 
            


        # diffuse albedo and normal
        for i in [0,3]:
            # concat the output of following layers:
            # Conv5/MidConv2    Conv4/DeConv0   Conv3/DeConv1   Conv2/DeConv2   Conv1/DeConv3   Conv0/DeConv4
            # concatenated outputs are fed as the input of the succedent layers

            var['Merge0_ch{}'.format(i)]=self.Concat([ var['Conv5_ch{}'.format(i)], var['MidConv2_ch{}'.format(i)] ], name='Merge0_ch{}'.format(i)) # 8*8*(256+256) -> 8*8*512

            var['DeConv0_ch{}'.format(i)]=self.DeConvDoubleResolutionBilinear(var['Merge0_ch{}'.format(i)], [kh, kw, nFilterFirstConv*32, nFilterFirstConv*16], name='DeConv0_ch{}'.format(i))  # 8*8*512 -> 16*16*256

            var['Merge1_ch{}'.format(i)]=self.Concat([ var['Conv4_ch{}'.format(i)], var['DeConv0_ch{}'.format(i)] ], name='Merge1_ch{}'.format(i)) # 16*16*(256+256) -> 16*16*512

            var['DeConv1_ch{}'.format(i)]=self.DeConvDoubleResolutionBilinear(var['Merge1_ch{}'.format(i)], [kh, kw, nFilterFirstConv*32, nFilterFirstConv*8], name='DeConv1_ch{}'.format(i))  # 16*16*512 -> 32*32*128
 
            var['Merge2_ch{}'.format(i)]=self.Concat([ var['Conv3_ch{}'.format(i)], var['DeConv1_ch{}'.format(i)] ], name='Merge2_ch{}'.format(i)) # 32*32*(128+128) -> 32*32*256

            var['DeConv2_ch{}'.format(i)]=self.DeConvDoubleResolutionBilinear(var['Merge2_ch{}'.format(i)], [kh, kw, nFilterFirstConv*16, nFilterFirstConv*4], name='DeConv2_ch{}'.format(i))  # 32*32*256 -> 64*64*64
 
            var['Merge3_ch{}'.format(i)]=self.Concat([ var['Conv2_ch{}'.format(i)], var['DeConv2_ch{}'.format(i)] ], name='Merge3ch{}'.format(i)) # 64*64*(64+64) -> 64*64*128

            var['DeConv3_ch{}'.format(i)]=self.DeConvDoubleResolutionBilinear(var['Merge3_ch{}'.format(i)], [kh, kw, nFilterFirstConv*8, nFilterFirstConv*2], name='DeConv3_ch{}'.format(i))  # 64*64*128 -> 128*128*32

            var['Merge4_ch{}'.format(i)]=self.Concat([ var['Conv1_ch{}'.format(i)], var['DeConv3_ch{}'.format(i)] ], name='Merge4_ch{}'.format(i)) # 128*128*(32+32) -> 128*128*64

            var['DeConv4_ch{}'.format(i)]=self.DeConvDoubleResolutionBilinear(var['Merge4_ch{}'.format(i)], [kh, kw, nFilterFirstConv*4, nFilterFirstConv], name='DeConv4_ch{}'.format(i))  # 128*128*64 -> 256*256*16
 
            var['Merge5_ch{}'.format(i)]=self.Concat([ var['Conv0_ch{}'.format(i)], var['DeConv4_ch{}'.format(i)] ], name='Merge5_ch{}'.format(i)) # 256*256*(16+16) -> 256*256*32
            

            # two final convolutions and sigmond 
            var['ConvFinal_ch{}'.format(i)]=self.ConvSameResolution(var['Merge5_ch{}'.format(i)], [kh, kw, nFilterFirstConv*2, nFilterFirstConv], name='ConvFinal_ch{}'.format(i))  # 256*256*32 -> 256*256*16
            
        self.predict_albedo=self.ConvSameResolutionSigmoid(var['ConvFinal_ch0'], [kw, kw, nFilterFirstConv, 3], name='ConvSigmoid_ch0')   # 256*256*16 -> 256*256*3   
        self.predict_normal=self.ConvSameResolutionSigmoid(var['ConvFinal_ch3'], [kw, kh, nFilterFirstConv, 3], name='ConvSigmoid_ch3')   # 256*256*16 -> 256*256*3   
        
        self.loss_albedo=self.MSELoss(self.data_albedo, self.predict_albedo, 'Loss_ch0')
        self.loss_normal=self.MSELoss(self.data_normal, self.predict_normal, 'Loss_ch3')

           

        # specular albedo and roughness
        for i in [1,2]:
            var['FCReLU0_ch{}'.format(i)]=self.FC(var['MidConv2_ch{}'.format(i)], nFisrtFC, name='FCReLU0_ch{}'.format(i))  # 8*8*256 -> 1024
            var['FCReLU1_ch{}'.format(i)]=self.FC(var['FCReLU0_ch{}'.format(i)], nFisrtFC//2, name='FCReLU1_ch{}'.format(i)) # 1024 -> 512

                
        self.predict_spec=self.FC(var['FCReLU1_ch1'.format(i)], 3, withReLU=False, name='FCFinal_ch1'.format(i))
        self.predict_roughness=self.FC(var['FCReLU1_ch2'.format(i)], 1, withReLU=False, name='FCFinal_ch2'.format(i))
       
        self.loss_spec=self.MSELoss(self.data_spec, self.predict_spec)
        self.loss_roughness=self.MSELoss(self.data_roughness, self.predict_roughness)

        self.loss=lossweight_d*self.loss_albedo+lossweight_s*self.loss_spec+lossweight_r*self.loss_roughness+lossweight_n*self.loss_normal
         


        
