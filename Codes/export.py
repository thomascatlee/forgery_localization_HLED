import os

import tensorflow as tf2
tf = tf2.compat.v1
tf.disable_v2_behavior()

config=tf.ConfigProto()
config.allow_soft_placement=True
config.log_device_placement=False
config.gpu_options.allow_growth=True
#config.gpu_options.per_process_gpu_memory_fraction = 0.4\

graph = tf.get_default_graph()
with tf.Session(config=config) as sess:
    loader = tf.train.import_meta_graph("../model/final_model_nist.ckpt.meta")
    loader.restore(sess, '../model/final_model_nist.ckpt')

    input_tf0 = graph.get_tensor_by_name('Placeholder:0')
    input_tf1 = graph.get_tensor_by_name('Placeholder_1:0')
    input_tf2 = graph.get_tensor_by_name('Placeholder_2:0')
    output_tf1 = graph.get_tensor_by_name('ArgMax_5:0')
    output_tf2 = graph.get_tensor_by_name('Softmax:0')
    output_tf3 = graph.get_tensor_by_name('ArgMax_2:0')

    sig_def = tf.saved_model.predict_signature_def(
        inputs={"input_layer": input_tf0,
                "y":input_tf1,
                "freqFeat": input_tf2},
        outputs={'final_predictions': output_tf1,
                 'final_probabilities': output_tf2,
                 'y2': output_tf3})        # Export checkpoint to SavedModel
    builder = tf.saved_model.builder.SavedModelBuilder('../outputpb')
    builder.add_meta_graph_and_variables(sess, [tf.saved_model.SERVING], strip_default_attrs=True, signature_def_map = {'predict': sig_def})
    builder.save()

os.system("python -m tf2onnx.convert --saved-model ../outputpb/ --output onnx_model.onnx --opset 9")










 
