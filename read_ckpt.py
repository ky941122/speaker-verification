from tensorflow.python import pywrap_tensorflow
import sys
import tensorflow as tf
import functools



if __name__ == "__main__":
    args = sys.argv
    checkpoint_path = args[1]

    graph = tf.Graph()
    with graph.as_default():
        session_conf = tf.ConfigProto(
            allow_soft_placement=True,
            log_device_placement=False)
        session_conf.gpu_options.allow_growth = True
        sess = tf.Session(config=session_conf)
        with sess.as_default():
            saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_path))


            def size(v):
                return functools.reduce(lambda x, y: x * y, v.get_shape().as_list())


            print("Trainable variables")
            for v in tf.trainable_variables():
                print("  %s, %s, %s, %s" % (v.name, v.device, str(v.get_shape()), size(v)))
            print("Total model size: %d" % (sum(size(v) for v in tf.trainable_variables())))





    # reader = pywrap_tensorflow.NewCheckpointReader(checkpoint_path)
    # var_to_shape_map = reader.get_variable_to_shape_map()
    # for key in var_to_shape_map:
    #     print("tensor_name: ", key)
    #     # print(reader.get_tensor(key)) # Remove this is you want to print only variable names