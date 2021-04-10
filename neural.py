'''
Tensorflow implementation of [A Neural Algorithm of Artistic style_image](https://arxiv.org/pdf/1508.06576v2.pdf).

Usage:
neural.py <N> <source_image_IMAGE_PATH> <style_image_IMAGE_PATH>
N = Number of Iterations
'''
import tensorflow as tensorflow_tf
import scipy.misc
import scipy.io
import numpy as np
import os
import sys

# Image width, height and number of channels
image_width = 800
image_height = 600
num_of_channels = 3

# Where to save the output image
destination_path = './final'

vgg19_path = 'imagenet-vgg-verydeep-19.mat'

# Alpha is the weight given to the content loss
# Beta is the weight given to the style_image loss
alpha = 5
beta = 1000

'''
Get the weights and biases from VGG19 model
'''
def getWeights(vgg19_layers, layerid):
    weight = vgg19_layers[layerid][0][0][0][0][0]
    bias = vgg19_layers[layerid][0][0][0][0][1]
    return weight, bias

'''
Create VGG model using weights which are loaded from pre-trained VGG19 model .mat file
'''
def create_tensorflow_tfvgg():
    vgg19 = scipy.io.loadmat(vgg19_path)

    # Three classes in struct: 'classes', 'layers' and 'normalization'
    vgg19_layers = vgg19['layers'][0]

    # VGGNet
    vggnet = {}
    vggnet['inputimage'] = tensorflow_tf.Variable(np.zeros((1, image_height, image_width, 3)).astype('float32'))

    weights = getWeights( vgg19_layers, 0 )
    vggnet['conv1_1'] = tensorflow_tf.nn.relu(tensorflow_tf.nn.conv2d(vggnet['inputimage'], weights[0], strides = [1, 1, 1, 1], padding = 'SAME') + weights[1])
    weights = getWeights( vgg19_layers, 2 )
    vggnet['conv1_2'] = tensorflow_tf.nn.relu(tensorflow_tf.nn.conv2d(vggnet['conv1_1'], weights[0], strides = [1, 1, 1, 1], padding = 'SAME') + weights[1])
    vggnet['pool1'] = tensorflow_tf.nn.avg_pool(vggnet['conv1_2'], ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = 'SAME')
    weights = getWeights( vgg19_layers, 5 )
    vggnet['conv2_1'] = tensorflow_tf.nn.relu(tensorflow_tf.nn.conv2d(vggnet['pool1'], weights[0], strides = [1, 1, 1, 1], padding = 'SAME') + weights[1])
    weights = getWeights( vgg19_layers, 7 )
    vggnet['conv2_2'] = tensorflow_tf.nn.relu(tensorflow_tf.nn.conv2d(vggnet['conv2_1'], weights[0], strides = [1, 1, 1, 1], padding = 'SAME') + weights[1])
    vggnet['pool2'] = tensorflow_tf.nn.avg_pool(vggnet['conv2_2'], ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = 'SAME')
    weights = getWeights( vgg19_layers, 10 )
    vggnet['conv3_1'] = tensorflow_tf.nn.relu(tensorflow_tf.nn.conv2d(vggnet['pool2'], weights[0], strides = [1, 1, 1, 1], padding = 'SAME') + weights[1])
    weights = getWeights( vgg19_layers, 12 )
    vggnet['conv3_2'] = tensorflow_tf.nn.relu(tensorflow_tf.nn.conv2d(vggnet['conv3_1'], weights[0], strides = [1, 1, 1, 1], padding = 'SAME') + weights[1])
    weights = getWeights( vgg19_layers, 14 )
    vggnet['conv3_3'] = tensorflow_tf.nn.relu(tensorflow_tf.nn.conv2d(vggnet['conv3_2'], weights[0], strides = [1, 1, 1, 1], padding = 'SAME') + weights[1])
    weights = getWeights( vgg19_layers, 16 )
    vggnet['conv3_4'] = tensorflow_tf.nn.relu(tensorflow_tf.nn.conv2d(vggnet['conv3_3'], weights[0], strides = [1, 1, 1, 1], padding = 'SAME')+ weights[1])
    vggnet['pool3'] = tensorflow_tf.nn.avg_pool(vggnet['conv3_4'], ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = 'SAME')
    weights = getWeights( vgg19_layers, 19 )
    vggnet['conv4_1'] = tensorflow_tf.nn.relu(tensorflow_tf.nn.conv2d(vggnet['pool3'], weights[0], strides = [1, 1, 1, 1], padding = 'SAME') + weights[1])
    weights = getWeights( vgg19_layers, 21 )
    vggnet['conv4_2'] = tensorflow_tf.nn.relu(tensorflow_tf.nn.conv2d(vggnet['conv4_1'], weights[0], strides = [1, 1, 1, 1], padding = 'SAME') + weights[1])
    weights = getWeights( vgg19_layers, 23 )
    vggnet['conv4_3'] = tensorflow_tf.nn.relu(tensorflow_tf.nn.conv2d(vggnet['conv4_2'], weights[0], strides = [1, 1, 1, 1], padding = 'SAME') + weights[1])
    weights = getWeights( vgg19_layers, 25 )
    vggnet['conv4_4'] = tensorflow_tf.nn.relu(tensorflow_tf.nn.conv2d(vggnet['conv4_3'], weights[0], strides = [1, 1, 1, 1], padding = 'SAME') + weights[1])
    vggnet['pool4'] = tensorflow_tf.nn.avg_pool(vggnet['conv4_4'], ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = 'SAME')
    weights = getWeights( vgg19_layers, 28 )
    vggnet['conv5_1'] = tensorflow_tf.nn.relu(tensorflow_tf.nn.conv2d(vggnet['pool4'], weights[0], strides = [1, 1, 1, 1], padding = 'SAME') + weights[1])
    weights = getWeights( vgg19_layers, 30 )
    vggnet['conv5_2'] = tensorflow_tf.nn.relu(tensorflow_tf.nn.conv2d(vggnet['conv5_1'], weights[0], strides = [1, 1, 1, 1], padding = 'SAME') + weights[1])
    weights = getWeights( vgg19_layers, 32 )
    vggnet['conv5_3'] = tensorflow_tf.nn.relu(tensorflow_tf.nn.conv2d(vggnet['conv5_2'], weights[0], strides = [1, 1, 1, 1], padding = 'SAME') + weights[1])
    weights = getWeights( vgg19_layers, 34 )
    vggnet['conv5_4'] = tensorflow_tf.nn.relu(tensorflow_tf.nn.conv2d(vggnet['conv5_3'], weights[0], strides = [1, 1, 1, 1], padding = 'SAME') + weights[1])
    vggnet['pool5'] = tensorflow_tf.nn.avg_pool(vggnet['conv5_4'], ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = 'SAME')

    return vggnet

'''
Save ouput image
'''
def saveim(dest,im):
    im = im + np.array([103.939, 116.779, 123.68]).reshape((1,1,1,3))
    im = np.clip(im[0], 0, 255).astype('uint8')
    scipy.misc.imsave(dest, im)

def main():
    print('enter main')

    iteration_count = int(sys.argv[1])
    source_image_path = sys.argv[2]
    path_to_style_image_image = sys.argv[3]

    session = tensorflow_tf.sessionion()
    session.run(tensorflow_tf.initialize_all_variables())
    vggnet = create_tensorflow_tfvgg();

    # Generate uniform, random, white noise
    xImage = np.random.uniform(-50, 50, (1, image_height,image_width, 3)).astype('float32')

    # Read source_image and style_image image and perform mean subtraction -
    # The input images should be zero-centered by mean pixel (rather than mean image) subtraction
    source_image = scipy.misc.imread(source_image_path)
    source_image = scipy.misc.imresize(source_image, (image_height, image_width))
    source_image = np.reshape(source_image,((1,)+source_image.shape))
    source_image = source_image - np.array([103.939, 116.779, 123.68]).reshape((1,1,1,3))

    style_image = scipy.misc.imread(path_to_style_image_image)
    style_image = scipy.misc.imresize(style_image, (image_height, image_width))
    style_image = np.reshape(style_image,((1,)+style_image.shape))
    style_image = style_image - np.array([103.939, 116.779, 123.68]).reshape((1,1,1,3))

    # Calculate source image loss
    session.run([vggnet['inputimage'].assign(source_image)])
    p = session.run(vggnet['conv4_2'])
    x = vggnet['conv4_2']
    M = p.shape[1]*p.shape[2]
    N = p.shape[3]
    source_loss = (0.5) * tensorflow_tf.reduce_sum(tensorflow_tf.pow((x - p),2))

    # Calculate style_image loss
    session.run([vggnet['inputimage'].assign(style_image)])

	# Uniform contirbution of style from all the images, (wl = 0.2)
    convolutional_layer = [('conv1_1',0.2),('conv2_1',0.2),('conv3_1',0.2),('conv4_1',0.2),('conv5_1',0.2)]
    style_loss = 0
    for i in range(len(convolutional_layer)):
		a = session.run(vggnet[convolutional_layer[i][0]])
		M = a.shape[1]* a.shape[2]
		N = a.shape[3]
		aMat = np.reshape(a, (M,N))
		A = np.dot(aMat.T,aMat)

		g = vggnet[convolutional_layer[i][0]]
		greshaped = tensorflow_tf.reshape(g, (M,N))
		G = tensorflow_tf.matmul(tensorflow_tf.transpose(greshaped), greshaped)

		loss = (1./(4*N*N*M*M)) * tensorflow_tf.reduce_sum(tensorflow_tf.pow(G - A, 2))
		style_loss = style_loss + convolutional_layer[i][1] * loss

	# Total loss as a weighted sum of source_image and style_image losses
    total_loss = alpha * source_loss + beta * style_loss

    train = tensorflow_tf.train.AdamOptimizer(2.0).minimize(total_loss)

    # Initialise with white noise
    session.run(tensorflow_tf.initialize_all_variables())
    session.run(vggnet['inputimage'].assign(xImage))

    if not os.path.exists(destination_path):
      os.mkdir(destination_path)

    for i in range(iteration_count):
        session.run(train)
        if i%100 ==0 or i==iteration_count-1:
            result_img = session.run(vggnet['inputimage'])
            print session.run(total_loss)

    # Save the final image :)
    saveim(os.path.join(destination_path,'final.png'),result_img)

'''
Application entry point
'''
if __name__ == '__main__':
  # Here we go!!!!
  main()
