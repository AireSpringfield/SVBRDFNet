import os, sys
working_path = os.path.dirname(os.path.realpath(__file__))
root_path = os.path.dirname(working_path)
sys.path.append(root_path + r'/Render')
sys.path.append(root_path + r'/Utils')

from multiprocessing import Process
from configparser import SafeConfigParser
import numpy as np
import tensorflow as tf
import cv2
from NetClass import SVBRDFNet
from utils import load_pfm, save_pfm, toHDR, toLDR



os.chdir(working_path)


def loadParams(filePath):
    params={}   

    config = SafeConfigParser()
    config.read(filePath)
    

    # [path]
    params['outFolder'] = config.get('path', 'outFolder')
    params['envMapFolder'] = config.get('path', 'envMapFolder')
    params['geometryPath'] = config.get('path', 'geometryPath')

    return params


def renderRelighting(renderer, albedo, spec, roughness, normal):

    renderer.cudacontext.push()

    renderer.SetPointLight(0, 0.27, -0.25, 1, 0, 0.6, 0.6, 0.6)
    renderer.SetAlbedoMap(albedo)
    renderer.SetSpecValue(spec)
    renderer.SetRoughnessValue(roughness)

    normal = normal * 2.0 - 1.0
    normal[0] = normal[0] * 2.5
    len = np.linalg.norm(normal, axis = 2)
    normal = normal / np.dstack((len, len, len))
    normal = 0.5*(normal + 1.0)

    renderer.SetNormalMap(normal*2.0 - 1.0)
    img = renderer.Render()

    renderer.SetEnvLightByID(43, 30, -10.0)
    renderer.SetAlbedoMap(albedo)
    renderer.SetSpecValue(spec)
    renderer.SetRoughnessValue(roughness)
    renderer.SetNormalMap(normal*2.0 - 1.0)
    img_1 = renderer.Render()

    renderer.cudacontext.pop()

    return 1.2 * img + 0.8 * img_1




if __name__ == '__main__':
    # python3 Test.py $modelFile$ $testSetPath(list.txt)$ $GPUid$
    modelFile= sys.argv[1]
    testSetPath = sys.argv[2]
    gpuid = int(sys.argv[3])

    params = loadParams(working_path + r'/Config.ini')

  


    # init fast online renderer
    onlineRender = FastRenderEngine(gpuid)
    onlineRender.SetGeometry('Plane')
    onlineRender.SetSampleCount(128, 512)
    onlineRender.PreLoadAllLight(r'{}/light.txt'.format(params['envMapFolder']))
    fovRadian = 60.0 / 180.0 * np.pi
    cameraDist = 1.0 / (math.tan(fovRadian / 2.0))
    onlineRender.SetCamera(0, 0, cameraDist, 0, 0, 0, 0, 1, 0, fovRadian, 0.01, 100, 256, 256)


    # init tensorflow network
    config = tf.ConfigProto()
    config.gpu_options.visible_device_list='{}'.format(gpuid)
    #config.gpu_options.allow_growth = True
    net = SVBRDFNet(config)
    net.CreateNet()

    saver=tf.train.Saver()
    saver.restore(net.sess, modelFile)


    path, file = os.path.split(testSetPath)
    with open(testSetPath, 'r') as f:
        filenames = f.read().strip().split('\n')
    np.random.shuffle(filenames)

    pixelCnt = 256*256





    for f in filenames:
        fullpath = path + r'/{}.jpg'.format(f.strip())
        data_id = '{}'.format(f.strip())
        print('Test {}\n'.format(f.strip()))
        img = toHDR(cv2.imread(fullpath))

        img_in = np.zeros((16,256,256,3))
        img_in[6,:,:,:] = img

        predict_a, predict_s, predict_r, predict_n = net.sess.run([net.predict_albedo, net.predict_spec, net.predict_roughness, net.predict_normal],
                    {net.is_training:False, net.data_image:img_in})
        predict_a = predict_a[6]
        predict_n = predict_n[6]
        predict_r = predict_r[6]
        predict_s = predict_s[6]

        factor = 0.5/np.mean(np.linalg.norm(predict_a, axis=2))
        predict_a = predict_a*factor
        predict_s = predict_s*factor

        predict_s = np.exp(predict_s)[np.newaxis, np.newaxis, :]*np.ones([256,256,3])
        predict_r = np.exp(predict_r)*np.ones([256, 256, 3])

        relighting = renderRelighting(onlineRender, predict_a, predict_s, predict_r, predict_n)

        cv2.imwrite(path+ r'/{}_albedo_fit.jpg'.format(data_id), toLDR(predict_a))
        cv2.imwrite(path+ r'/{}_normal_fit.jpg'.format(data_id), toLDR(predict_n))
        cv2.imwrite(path+ r'/{}_specalbedo_fit.jpg'.format(data_id), toLDR(predict_s))
        cv2.imwrite(path+ r'/{}_roughness_fit.jpg'.format(data_id), toLDR(predict_r))
        cv2.imwrite(path+ r'{}_relighting.jpg'.format(data_id), toLDR(relighting))
        
