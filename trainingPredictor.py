# import the necessary packages
import multiprocessing
import dlib
import time

t = time.time()
options = dlib.shape_predictor_training_options()
options.tree_depth = 4
options.nu = 0.1
options.cascade_depth = 15
options.feature_pool_size = 400
options.num_test_splits = 50
options.oversampling_amount = 5
options.oversampling_translation_jitter = 0.1
options.be_verbose = True
options.num_threads = multiprocessing.cpu_count()
print("Wait for the predictor to be trained...")
dlib.train_shape_predictor("F:/dataset/ibug_300W_large_face_landmark_dataset/datasetLips.xml",
                           "F:/dataset/ibug_300W_large_face_landmark_dataset/lipDetectorSpeed.dat", options)
print("time taken to train : {:.3f}".format(time.time() - t))

#print("Wait for the predictor to be trained...\nTraining with cascade depth: 18\nTraining with tree depth: 5\nTraining with 500 trees per cascade level.\nTraining with nu: 0.1\nTraining with random seed: \nTraining with oversampling amount: 5\nTraining with oversampling translation jitter: 0.1\nTraining with landmark_relative_padding_mode: 1\nTraining with feature pool size: 500\nTraining with feature pool region padding: 0\nTraining with 4 threads.\nTraining with lambda_param: 0.1\nTraining with 75 split tests.\nFitting trees...\nTraining complete\nTraining complete, saved predictor to file F:/dataset/ibug_300W_large_face_landmark_dataset/lipDetectorSpeed.dat\ntime taken to train : 3422.751")
