
import os, sys, pickle, time, random, math
import numpy as np
import tensorflow as tf
import cv2

from configparser import SafeConfigParser
from multiprocessing import Process, Pipe



file_path=os.path.realpath(__file__)
working_path=os.path.dirname(file_path)
root_path=os.path.dirname(working_path)
sys.path.append(root_path+r'/Render')   # add 'r' to prevent escape
sys.path.append(root_path+r'/Utils')


os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'



from utils import make_dir, autoExposure, normalBatchToThetaPhiBatch, renormalize, DataLoaderSVBRDF, RealDataLoaderSVBRDF, normalizeAlbedoSpec, save_pfm
from NetClass import SVBRDFNet
from FastRendererCUDA import FastRenderEngine








# load params from SVBRDF_Net_Config.ini




def loadParams(filePath):
    params={}   

    # entries in 'DEFAULT' section can be retrieved by get method from any exist section
    config = SafeConfigParser({'albedoWeight':'1.0', 
                               'specWeight':'1.0', 
                               'roughnessWeight':'1.0', 
                               'normalWeight':'1.0',
                               'NetworkFile':'',
                               'loopRestartFrequency':'-1', 
                               'PreTrainSpecNet':'', 
                               'grayLight':'0', 
                               'normalizeAlbedo':'0', 
                               'PreTrainSpecNetNoLoop':'', 
                               'LogSpec':'0', 
                               'autoExposure':'0', 
                               'autoExposureLUTFile':'', 
                               'lightPoolFile':'', 
                               'NormalLoss':'L2'})
    config.read(filePath)
    

    # [path]
    params['outFolder'] = config.get('path', 'outFolder')
    params['envMapFolder'] = config.get('path', 'envMapFolder')
    params['geometryPath'] = config.get('path', 'geometryPath')

    

    # [device]
    params['randomSeed'] = config.getint('device', 'randomSeed')

    # [solver]
   
    params['lr'] = config.getfloat('solver', 'lr')
    params['lrDecay'] = config.getfloat('solver', 'lrDecay')
    params['batchSize'] = config.getint('solver', 'batchSize')
    params['autoExposure'] = config.getint('solver', 'autoExposure')

    # [stopping] 
    params['nMaxEpoch'] = config.getint('stopping', 'nMaxEpoch')
    params['nMaxIter'] = config.getint('stopping', 'nMaxIter')

    # [loop]
    params['autoLoopRatio'] = config.getboolean('loop', 'autoLoopRatio')

    #the SA training would alternating between 'normalBatchLength' iteration of normal training and 'loopBatchLength' of self-augment training 
    params['normalBatchLength'] = config.getint('loop', 'normalBatchLength')
    params['loopStartEpoch'] = config.getint('loop', 'loopStartEpoch')
    params['loopStartIteration'] = config.getint('loop', 'loopStartIteration')  #add loop after this number of normal training.
    params['loopBatchLength'] = config.getint('loop', 'loopBatchLength')        #how many mini-batch iteration for ever loop optimize


    # [network]
    params['Channal'] = config.get('network', 'Channal')
    params['LogRoughness'] = config.getboolean('network', 'LogRoughness')
    params['LogSpec'] = config.getboolean('network', 'LogSpec')
    params['BN'] = config.getboolean('network', 'BN')
    params['nFirstFeatureMap'] = config.getint('network', 'nFirstFeatureMap')
    params['LogRoughness'] = config.getboolean('network', 'LogRoughness')
    params['LogSpec'] = config.getboolean('network', 'LogSpec')
    params['albedoWeight'] = config.getfloat('network', 'albedoWeight')
    params['specWeight'] = config.getfloat('network', 'specWeight')
    params['roughnessWeight'] = config.getfloat('network', 'roughnessWeight')
    params['normalWeight'] = config.getfloat('network', 'normalWeight')


    # [dataset]
    params['dataset'] = config.get('dataset', 'dataset')
    params['testDataset'] = config.get('dataset', 'testDataset')
    params['unlabelDataset'] = config.get('dataset', 'unlabelDataset')
    params['LDR'] = config.getint('dataset', 'LDR')         
    params['grayLight'] = config.getboolean('dataset', 'grayLight')
    params['normalizeAlbedo'] = config.getboolean('dataset', 'normalizeAlbedo')
    params['lightPoolFile'] = config.get('dataset', 'lightPoolFile')
    params['autoExposureLUTFile'] = config.get('dataset', 'autoExposureLUTFile')

    # [checkpoint]
    params['logLossStepIteration'] = config.getint('checkpoint', 'logLossStepIteration')
    params['checkPointStepIteration'] = config.getint('checkpoint', 'checkPointStepIteration')
    params['checkPointStepEpoch'] = config.getint('checkpoint', 'checkPointStepEpoch')
    
  

    with open(params['envMapFolder']+r'/light.txt') as f:
        lightID = map(int, f.read().strip().split('\n'))
        params['lightID'] = list(np.array(list(lightID)) - 1) 

    
    if(params['lightPoolFile']!=''):
        params['lightPool']=pickle.load(open(params['envMapFolder']+r'/{}'.format(params['lightPoolFile']),'rb'), encoding='latin1')

    if(params['autoExposureLUTFile']!=''):
        params['autoExposureLUT']=pickle.load(open(params['envMapFolder']+r'/{}'.format(params['autoExposureLUTFile']), 'rb'), encoding='latin1')
   
    return params


def renderOnlineEnvlight(brdfBatch, onlineRender, params, lightIDs=[], lightXforms=[], lightNorms=[]):



    imgBatch = np.zeros((brdfBatch.shape[0], 256, 256, 3))



    if(lightIDs == []):
        lightIDs = random.sample(params['lightID'], brdfBatch.shape[0])
    if(lightXforms == []):
        angle_y = np.random.uniform(0.0, 360.0, brdfBatch.shape[0])
        angle_x = np.random.uniform(-45.0, 45.0, brdfBatch.shape[0])
    else:
        angle_y = lightXforms[1]
        angle_x = lightXforms[0]

  
    onlineRender.cudacontext.push()

    for i in range(0, brdfBatch.shape[0]):
        onlineRender.SetEnvLightByID(lightIDs[i] + 1)
        onlineRender.SetLightXform(angle_x[i], angle_y[i])

        onlineRender.SetAlbedoMap(brdfBatch[i, :, :, 0:3])
        onlineRender.SetSpecValue(brdfBatch[i, 0, 0, 3:6])
        onlineRender.SetRoughnessValue(brdfBatch[i, 0, 0, 6])

          
        onlineRender.SetNormalMap(2.0 * brdfBatch[i, :, :, 7:10] - 1.0)
        imgBatch[i, :, :, :] = onlineRender.Render()

        if(params['autoExposure'] == 1):
            imgBatch[i, :, :, 0] = imgBatch[i, :, :, 0] / lightNorms[i][0]
            imgBatch[i, :, :, 1] = imgBatch[i, :, :, 1] / lightNorms[i][1]
            imgBatch[i, :, :, 2] = imgBatch[i, :, :, 2] / lightNorms[i][2]
        elif(params['autoExposure'] == 2):
            onlineRender.SetAlbedoValue([1.0, 1.0, 1.0])
            onlineRender.SetSpecValue([0.0, 0.0, 0.0])
            normal_one = np.dstack((np.ones((256, 256)), np.zeros((256, 256)), np.zeros((256, 256))))
            onlineRender.SetNormalMap(normal_one)
            img_norm = onlineRender.Render()
            normValue = np.mean(img_norm, axis=(0, 1))
            imgBatch[i, 0, :, :] = imgBatch[i, 0, :, :] / normValue[0]
            imgBatch[i, 1, :, :] = imgBatch[i, 1, :, :] / normValue[1]
            imgBatch[i, 2, :, :] = imgBatch[i, 2, :, :] / normValue[2]
            # autoExposure(imgBatch[i,:,:,:])
        imgBatch[i, :, :, :] = 0.5 * imgBatch[i, :, :, :]
      
    
    onlineRender.cudacontext.pop()

    return imgBatch




if __name__ == '__main__':
    # python3 Train.py Config.ini $RESTORE_TAG$ $GPUID$ $RENDERGPUID$ (optional)$RESTORE_OUT_FOLDER 
    os.chdir(working_path)
   
    configFilePath = sys.argv[1]
    restoreTraining = int(sys.argv[2])
    gpuid = int(sys.argv[3])
    rendergpuid = int(sys.argv[4])


    params=loadParams(configFilePath)
    '''
    if(gpuid == rendergpuid):
        os.environ['CUDA_VISIBLE_DEVICES']= '{}'.format(gpuid) 
        params['rendergpuid'] = 0
    else:
        os.environ['CUDA_VISIBLE_DEVICES']= '{},{}'.format(gpuid, rendergpuid)  # reenumerate to 0,1
        params['rendergpuid'] = 1
    '''
    #os.environ['CUDA_VISIBLE_DEVICES']='{}'.format(rendergpuid)
    #params['rendergpuid']=0

    params['rendergpuid'] = rendergpuid
    
    date=time.strftime(r'%Y-%m-%d_%H:%M:%S')
    outFolder = params['outFolder'] + r'/{}'.format(date)
    if(restoreTraining):
        if(len(sys.argv)==6):
            outFolder = sys.argv[5]
        else:
            raise InputError('Must assign the output folder of the checkpoint!')


    logFolder = outFolder + r'/Summary'
    modelFolder = outFolder + r'/Models'
    checkpoint_model = modelFolder + r'/model-checkpoint.ckpt'
    checkpoint_status = modelFolder + r'/status-checkpoint.txt'
    final_model = modelFolder + r'/model-final.ckpt'
    final_status = modelFolder + r'/status-final.txt'               


    make_dir(outFolder)
    make_dir(logFolder)
    make_dir(modelFolder)

    # start loading data for train 

    def DataLoadProcess(pipe, datafile, params, isTest = False):
        path, file = os.path.split(datafile)
        batchSize = 1 if isTest else params['batchSize']
        dataset = DataLoaderSVBRDF(path, file, 384, 384, not isTest)
        dataset.shuffle(params['randomSeed'])
        pipe.send(dataset.dataSize)
        counter = 0
        posInDataSet = 0
        epoch = 0
  
        if(params['LDR'] == 1):
            dataset.ldr = True       

        while(True):
            imgbatch, brdfbatch, name = dataset.GetBatchWithName(posInDataSet, batchSize)
          
            
            if(params['normalizeAlbedo']):
                brdfbatch = normalizeAlbedoSpec(brdfbatch)
                
            imgbatch = imgbatch.transpose([0,2,3,1])
            brdfbatch = brdfbatch.transpose([0,2,3,1])

            pipe.send((imgbatch, brdfbatch, name))

            counter = counter + batchSize
            posInDataSet = (posInDataSet + batchSize) % dataset.dataSize
            newepoch = counter / dataset.dataSize
            if(newepoch != epoch):
                dataset.shuffle()
            epoch = newepoch

    def RealUnlabelDataLoadProcess(pipe, datafile, params):
        path, file = os.path.split(datafile)
        batchSize = params['batchSize']
        dataset = RealDataLoaderSVBRDF(path, file)

        dataset.shuffle(params['randomSeed'])


        pipe.send(dataset.dataSize)

        counter = 0
        posInDataSet = 0
        epoch = 0

        while(True):
            imgbatch = dataset.GetBatch(posInDataSet, batchSize)
            for i in range(0, batchSize):
                imgbatch[i, :, :, :] = autoExposure(imgbatch[i, :, :, :])

            imgbatch = imgbatch.transpose([0,2,3,1])
            pipe.send(imgbatch)
            counter = counter + batchSize
            posInDataSet = (posInDataSet + batchSize) % dataset.dataSize
            newepoch = counter / dataset.dataSize
            if(newepoch != epoch):
                dataset.shuffle()
            epoch = newepoch



    # init fast online renderer
    onlineRender = FastRenderEngine(params['rendergpuid'])
    onlineRender.SetGeometry('Plane')
    onlineRender.SetSampleCount(128, 512)
    onlineRender.PreLoadAllLight(r'{}/light.txt'.format(params['envMapFolder']))
    fovRadian = 60.0 / 180.0 * np.pi
    cameraDist = 1.0 / (math.tan(fovRadian / 2.0))
    onlineRender.SetCamera(0, 0, cameraDist, 0, 0, 0, 0, 1, 0, fovRadian, 0.01, 100, 256, 256)



    # set random seeds
    random.seed(params['randomSeed'])
    tf.set_random_seed(params['randomSeed'])
    np.random.seed(params['randomSeed'])


    # init tensorflow network
    config = tf.ConfigProto()
    config.gpu_options.visible_device_list='{}'.format(gpuid)
    #config.gpu_options.allow_growth = True
    net = SVBRDFNet(config)
    net.CreateNet()

        # config Adam solver
    

    lr=tf.train.inverse_time_decay(params['lr'], net.total_iter, 1, params['lrDecay'])


    with tf.name_scope('optimizier'):
        optimizer = tf.train.AdamOptimizer(lr).minimize(net.loss, global_step=net.total_iter)

    # config summary, saver, writer
    summary_lr = tf.summary.scalar('learning_rate', lr)

    with tf.name_scope('summary_train'):
        loss_albedo_train = tf.summary.scalar('loss_albedo_train', net.loss_albedo)
        loss_spec_train = tf.summary.scalar('loss_spec_train',net.loss_spec)
        loss_roughness_train = tf.summary.scalar('loss_roughness_train', net.loss_roughness)
        loss_normal_train = tf.summary.scalar('loss_normal_train', net.loss_normal)
        loss_total_train = tf.summary.scalar('loss_total_train', net.loss)

    with tf.name_scope('summary_test'):
        loss_albedo_test = tf.summary.scalar('loss_albedo_test', net.loss_albedo)
        loss_spec_test = tf.summary.scalar('loss_spec_test',net.loss_spec)
        loss_roughness_test = tf.summary.scalar('loss_roughness_test', net.loss_roughness)
        loss_normal_test = tf.summary.scalar('loss_normal_test', net.loss_normal)
        loss_total_test = tf.summary.scalar('loss_total_test', net.loss)

    summary_train = tf.summary.merge([summary_lr, loss_albedo_train, loss_spec_train, loss_roughness_train, loss_normal_train, loss_total_train])
    summary_test = tf.summary.merge([summary_lr, loss_albedo_test, loss_spec_test, loss_roughness_test, loss_normal_test, loss_total_test])
    summary_writer = tf.summary.FileWriter(logFolder, net.sess.graph)
    saver = tf.train.Saver()




   


    # import training data (both labeled and unlabeled)
    

    pipe_train_recv, pipe_train_send = Pipe(False)
    pipe_test_recv, pipe_test_send = Pipe(False)
    pipe_loop_recv, pipe_loop_send = Pipe(False)
    

    loader_train = Process(target = DataLoadProcess, args = (pipe_train_send, params['dataset'], params))
    loader_test = Process(target = DataLoadProcess, args = (pipe_test_send, params['testDataset'], params, True))
    loader_loop = Process(target = RealUnlabelDataLoadProcess, args = (pipe_loop_send, params['unlabelDataset'], params))

    
    loader_train.daemon = True
    loader_test.daemon = True
    loader_loop.daemon = True

    loader_train.start()
    loader_test.start()
    loader_loop.start()

    print('Start loading data...\n')
    print('Wait for 5 seconds to load some data...\n')
    time.sleep(5)
    
   
    dataSize = pipe_train_recv.recv()
    testDataSize = pipe_test_recv.recv()
    loopDataSize = pipe_loop_recv.recv()
 
    totalDataSize = dataSize + loopDataSize

    # setting some condition variables: iteration, epoch

    if(params['autoLoopRatio']):
        params['loopBatchLength'] = params['normalBatchLength'] * int(np.round(loopdatasize / dataSize))

    if(params['loopStartEpoch'] != -1):
       params['loopStartIteration'] = dataSize * params['loopStartEpoch'] // params['batchSize']


    train_iter = 0
    loop_iter = 0
    total_iter =0

    train_epoch = -1
    loop_epoch = -1

    posInDataset = 0
    posInUnlabelDataset = 0 

    if(restoreTraining == True):
        with open(checkpoint_status, 'r') as f:
            status = f.read().split()
            train_iter = int(status[1])
            loop_iter = int(status[3])
            total_iter = int(status[5])
            train_epoch = int(status[7])
            loop_epoch = int(status[9])
            posInDataset = int(status[11])
            posInUnlabelDataset = int(status[13])
        saver.restore(net.sess, checkpoint_model)

    
    # load pre-computed auto-exposure data
    lightPool = []
    lightNormPool = []

 
    if(params['lightPoolFile'] != '' and params['lightPool']):
        for m in params['lightPool']:
            for l in range(0, params['lightPool'][m].shape[0]): 
                for v in range(0, params['lightPool'][m].shape[1] - 1):
                    rotX = params['lightPool'][m][l,v,0]
                    rotY = params['lightPool'][m][l,v,1]
                    #strLightMat = 'r,0,1,0,{}/r,1,0,0,{}/end'.format(rotY, rotX)
                    lightPool.append((params['lightID'][l], (rotX, rotY)))


    if(params['autoExposureLUTFile'] != '' and params['autoExposure']):
        for m in params['autoExposureLUT']:
            for l in range(0, params['autoExposureLUT'][m].shape[0]):   # params['lightNormPool'][m].shape == (49, 10, 3)
                for v in range(0, params['autoExposureLUT'][m].shape[1] - 1):
                    norm = params['autoExposureLUT'][m][l,v]
                    lightNormPool.append(norm)





    




    print('Start training...\n')
    print('labeled data size: {}    loop data size: {}\n'.format(dataSize, loopDataSize) )
    net.sess.run(tf.global_variables_initializer())
  



    

    while(True):
        # labeled training
        for i in range(0, params['normalBatchLength']):

            if(total_iter == params['nMaxIter'] or train_epoch == params['nMaxEpoch']):
                break
            # load data from training set
          
            img_data, brdf_data, names = pipe_train_recv.recv()

            if(params['LogSpec']):
                brdf_data[:,:,:,3:6] = np.log(brdf_data[:,:,:,3:6]) # Note the order
            
            if(params['LogRoughness']):
                brdf_data[:,:,:,6:7] = np.log(brdf_data[:,:,:,6:7])

          

            # brdf_data:
            # 0:3 = diffuse albedo  3:6 = specular albedo   6:7 = roughness    7:10 = normal
            net.sess.run([optimizer], feed_dict = {net.is_training:True,  net.data_image:img_data, 
                net.data_albedo:brdf_data[:,:,:,0:3], 
                net.data_spec:brdf_data[:,0,0,3:6],
                net.data_roughness:brdf_data[:,0,0,6:7],
                net.data_normal:brdf_data[:,:,:,7:10]})

            ##############################################################
            # add summary
            if(total_iter % params['logLossStepIteration'] == 0):
                # summary_train
                [summary] = net.sess.run([summary_train], feed_dict = {net.is_training:False,  net.data_image:img_data, 
                    net.data_albedo:brdf_data[:,:,:,0:3], 
                    net.data_spec:brdf_data[:,0,0,3:6],
                    net.data_roughness:brdf_data[:,0,0,6:7],
                    net.data_normal:brdf_data[:,:,:,7:10]})

                summary_writer.add_summary(summary, total_iter)

                # summary_test
                img_data, brdf_data, names = pipe_test_recv.recv()
                if(params['LogSpec']):
                    brdf_data[:,:,:,3:6] = np.log(brdf_data[:,:,:,3:6]) # Note the order
                if(params['LogRoughness']):
                    brdf_data[:,:,:,6:7] = np.log(brdf_data[:,:,:,6:7]) 

                [summary] = net.sess.run([summary_test], feed_dict = {net.is_training:False,  net.data_image:img_data, 
                    net.data_albedo:brdf_data[:,:,:,0:3], 
                    net.data_spec:brdf_data[:,0,0,3:6],
                    net.data_roughness:brdf_data[:,0,0,6:7],
                    net.data_normal:brdf_data[:,:,:,7:10]})                

                summary_writer.add_summary(summary, total_iter)
            
            if(total_iter<=10):
                print('train_iter: {}   loop_iter: {}   total_iter: {}'.format(train_iter, loop_iter, total_iter))

            # and increase iter variables
            train_iter += 1
            total_iter += 1
            posInDataset = (posInDataset + params['batchSize']) % dataSize
            train_epoch = train_iter * params['batchSize'] // dataSize


                
       




        # self-augmentation training
        if(total_iter >= params['loopStartIteration']):
            height = width = 256
            nBRDFChannels = 10

            for k in range(0, params['loopBatchLength']):
                if(total_iter >= params['nMaxIter'] or train_epoch >= params['nMaxEpoch']):
                    break
                

                

                img_data = pipe_loop_recv.recv()
                predict_a, predict_s, predict_r, predict_n = net.sess.run([net.predict_albedo, net.predict_spec, net.predict_roughness, net.predict_normal],
                    {net.is_training:False, net.data_image:img_data})


                predict_brdf = np.zeros([params['batchSize'], height, width, nBRDFChannels])

                predict_brdf[:,:,:,0:3] = predict_a

                if(params['LogSpec']):
                    predict_brdf[:,:,:,3:6] = np.exp(predict_s)[:,np.newaxis, np.newaxis,:] * np.ones([params['batchSize'], height, width, 1]) # 'broadcast' 
                else:
                    predict_brdf[:,:,:,3:6] = predict_s[:,np.newaxis, np.newaxis,:] * np.ones([params['batchSize'], height, width, 1])
                
                if(params['LogRoughness']):
                    predict_brdf[:,:,:,6:7] = np.exp(predict_r)[:,np.newaxis,np.newaxis,:]*np.ones([params['batchSize'], height, width, 1])
                else:
                    predict_brdf[:,:,:,6:7] = predict_r[:,np.newaxis,np.newaxis,:]*np.ones([params['batchSize'], height, width, 1])

                predict_brdf[:,:,:,7:10] = predict_n

                # clamp
                predict_brdf[:,:,:,0:7] = np.minimum( np.maximum(predict_brdf[:,:,:,0:7], 0.001), 1.0)
                # normalize normal map
                predict_brdf[:,:,:,7:10]=renormalize(predict_brdf[:,:,:,7:10])


                #save_pfm(outFolder+r'/a image_a.pfm', predict_brdf[0, :, :, 0:3])
                #print('predict brdf saved to {}'.format(outFolder))

                # random select light from light pool
                xforms = [[], []]
                renderIds = []
                lightNorms = []

                if(params['lightPoolFile']!=''):
                    selectedIds = np.random.choice(len(lightPool), params['batchSize'], replace=False)
                    renderIds = [lightPool[i][0] for i in selectedIds]
                    for i in selectedIds:
                        xforms[0].append(lightPool[i][1][0])
                        xforms[1].append(lightPool[i][1][1])
                    if(params['autoExposureLUTFile'] != '' and params['autoExposure']):
                        lightNorms = [lightNormPool[i] for i in selectedIds]


                
                # render a image with the predicted BRDFs
                img_predict =renderOnlineEnvlight(predict_brdf, onlineRender, params, renderIds, xforms, lightNorms)
                # re-feeding and training
                if(params['LogSpec']):
                    predict_brdf[:, :, :, 3:6] = np.log(predict_brdf[:, :, :, 3:6])
                if(params['LogRoughness']):
                    predict_brdf[:, :, :, 6:7] = np.log(predict_brdf[:, :, :, 6:7])

                

                net.sess.run([optimizer], feed_dict = {net.is_training:True,  net.data_image: img_predict, 
                    net.data_albedo:predict_brdf[:,:,:,0:3], 
                    net.data_spec:predict_brdf[:,0,0,3:6],
                    net.data_roughness:predict_brdf[:,0,0,6:7],
                    net.data_normal:predict_brdf[:,:,:,7:10]})
                
                
                # add summary
                if(total_iter % params['logLossStepIteration'] == 0):
                    # summary_train
                    [summary] = net.sess.run([summary_train], feed_dict = {net.is_training:False,  net.data_image:img_data, 
                        net.data_albedo:brdf_data[:,:,:,0:3], 
                        net.data_spec:brdf_data[:,0,0,3:6],
                        net.data_roughness:brdf_data[:,0,0,6:7],
                        net.data_normal:brdf_data[:,:,:,7:10]})

                    summary_writer.add_summary(summary, total_iter)

                    # summary_test
                    img_data, brdf_data, names = pipe_test_recv.recv()
                    if(params['LogSpec']):
                        brdf_data[:,:,:,3:6] = np.log(brdf_data[:,:,:,3:6]) # Note the order
                    if(params['LogRoughness']):
                        brdf_data[:,:,:,6:7] = np.log(brdf_data[:,:,:,6:7]) 

                    [summary] = net.sess.run([summary_test], feed_dict = {net.is_training:False,  net.data_image:img_data, 
                        net.data_albedo:brdf_data[:,:,:,0:3], 
                        net.data_spec:brdf_data[:,0,0,3:6],
                        net.data_roughness:brdf_data[:,0,0,6:7],
                        net.data_normal:brdf_data[:,:,:,7:10]})                

                    summary_writer.add_summary(summary, total_iter)
                
                loop_iter += 1
                total_iter += 1
                posInUnlabelDataset = (posInUnlabelDataset + params['batchSize']) % loopDataSize
                loop_epoch = loop_iter * params['batchSize'] // loopDataSize




####################################################################

        # checkpoint

        if(total_iter % params['checkPointStepIteration'] == 0 and total_iter>0):
            #os.remove(checkpoint_model)
            saver.save(net.sess, checkpoint_model)
            print(r'checkpoint: model saved to {}   total_iter: {}'.format(checkpoint_model, total_iter))

            with open(checkpoint_status, 'w') as f:
                f.write('train_iter: {}\n'.format(train_iter))
                f.write('loop_iter: {}\n'.format(loop_iter))
                f.write('total_iter: {}\n'.format(total_iter))
                f.write('train_epoch: {}\n'.format(train_epoch))
                f.write('loop_epoch: {}\n'.format(loop_epoch))
                f.write('posInDataset: {}\n'.format(posInDataset))
                f.write('posInUnlabelDataset: {}\n'.format(posInUnlabelDataset))  
        
    

        
    


        # break conditions
        if(total_iter == params['nMaxIter'] or train_epoch == params['nMaxEpoch']):
           
            # save final model
            saver.save(net.sess, final_model)

            with open(final_status, 'w') as f:
                f.write('train_iter: {}\n'.format(train_iter))
                f.write('loop_iter: {}\n'.format(loop_iter))
                f.write('total_iter: {}\n'.format(total_iter))
                f.write('train_epoch: {}\n'.format(train_epoch))
                f.write('loop_epoch: {}\n'.format(loop_epoch))
                f.write('posInDataset: {}\n'.format(posInDataset))
                f.write('posInUnlabelDataset: {}\n'.format(posInUnlabelDataset))

            break
