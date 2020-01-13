# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

def setflags(clear=True):
    import collections
    import tensorflow as tf

    flags = tf.compat.v1.flags
    FLAGS = flags.FLAGS

    if clear:
        ## Existing flags cannot be Defined a second time and the way the Notebook runs 
        #  this file gets called multiple times. the following code attempts to restore the
        #  flags to the state they are when flags first gets called.
        FLAGS.remove_flag_values(list(FLAGS)) 

        flags.DEFINE_bool('alsologtostderr', False, '')
        flags.DEFINE_string('log_dir', '', '')
        flags.DEFINE_bool('logtostderr', False, '')
        flags.DEFINE_bool('only_check_args', False, '')
        flags.DEFINE_bool('op_conversion_fallback_to_while_loop', False, '')
        flags.DEFINE_bool('pdb_post_mortem', False, '')
        flags.DEFINE_bool('profile_file', None, '')
        flags.DEFINE_bool('run_with_pdb', False, '')
        flags.DEFINE_bool('run_with_profiling', False, '')
        flags.DEFINE_bool('showprefixforinfo', True, '')
        flags.DEFINE_string('stderrthreshold', 'fatal', '')
        flags.DEFINE_integer('test_random_seed', 301, '')
        flags.DEFINE_bool('test_randomize_ordering_seed', None, '')
        flags.DEFINE_string('test_srcdir', '', '')
        flags.DEFINE_string('test_tmpdir', '/tmp/absl_testing', '')
        flags.DEFINE_bool('use_cprofile_for_profiling', True, '')
        flags.DEFINE_integer('v', -1, '')
        flags.DEFINE_integer('verbosity', -1, '')
        flags.DEFINE_string('xml_output_file', '', '')

    ## There seems to be an API problem causing "UnrecognizedFlagError: Unknown command
    #  line flag 'f'" so I am adding a flag by that name
    #  see: https://stackoverflow.com/questions/48198770/tensorflow-1-5-0-rc0-error-using-tf-app-flags
    flags.DEFINE_string('f', None, 'See notes in code')


    flags.DEFINE_string(
        "bert_config_file", None,
        "The config json file corresponding to the pre-trained BERT model. "
        "This specifies the model architecture.")

    flags.DEFINE_string("vocab_file", None,
                        "The vocabulary file that the BERT model was trained on.")

    flags.DEFINE_string(
        "output_dir", None,
        "The output directory where the model checkpoints will be written.")

    flags.DEFINE_string("train_precomputed_file", None,
                        "Precomputed tf records for training.")

    flags.DEFINE_integer("train_num_precomputed", None,
                         "Number of precomputed tf records for training.")

    flags.DEFINE_string(
        "predict_file", None,
        "NQ json for predictions. E.g., dev-v1.1.jsonl.gz or test-v1.1.jsonl.gz")

    flags.DEFINE_string(
        "output_prediction_file", None,
        "Where to print predictions in NQ prediction format, to be passed to"
        "natural_questions.nq_eval.")

    flags.DEFINE_string(
        "init_checkpoint", None,
        "Initial checkpoint (usually from a pre-trained BERT model).")

    flags.DEFINE_bool(
        "do_lower_case", True,
        "Whether to lower case the input text. Should be True for uncased "
        "models and False for cased models.")

    flags.DEFINE_integer(
        "max_seq_length", 384,
        "The maximum total input sequence length after WordPiece tokenization. "
        "Sequences longer than this will be truncated, and sequences shorter "
        "than this will be padded.")

    flags.DEFINE_integer(
        "doc_stride", 128,
        "When splitting up a long document into chunks, how much stride to "
        "take between chunks.")

    flags.DEFINE_integer(
        "max_query_length", 64,
        "The maximum number of tokens for the question. Questions longer than "
        "this will be truncated to this length.")

    flags.DEFINE_bool("do_train", False, "Whether to run training.")

    flags.DEFINE_bool("do_predict", False, "Whether to run eval on the dev set.")

    flags.DEFINE_integer("train_batch_size", 32, "Total batch size for training.")

    flags.DEFINE_integer("predict_batch_size", 8,
                         "Total batch size for predictions.")

    flags.DEFINE_float("learning_rate", 5e-5, "The initial learning rate for Adam.")

    flags.DEFINE_float("num_train_epochs", 3.0,
                       "Total number of training epochs to perform.")

    flags.DEFINE_float(
        "warmup_proportion", 0.1,
        "Proportion of training to perform linear learning rate warmup for. "
        "E.g., 0.1 = 10% of training.")

    flags.DEFINE_integer("save_checkpoints_steps", 1000,
                         "How often to save the model checkpoint.")

    flags.DEFINE_integer("iterations_per_loop", 1000,
                         "How many steps to make in each estimator call.")

    flags.DEFINE_integer(
        "n_best_size", 20,
        "The total number of n-best predictions to generate in the "
        "nbest_predictions.json output file.")

    flags.DEFINE_integer(
        "max_answer_length", 30,
        "The maximum length of an answer that can be generated. This is needed "
        "because the start and end predictions are not conditioned on one another.")

    flags.DEFINE_float(
        "include_unknowns", -1.0,
        "If positive, probability of including answers of type `UNKNOWN`.")

    flags.DEFINE_bool("use_tpu", False, "Whether to use TPU or GPU/CPU.")

    flags.DEFINE_string(
        "tpu_name", None,
        "The Cloud TPU to use for training. This should be either the name "
        "used when creating the Cloud TPU, or a grpc://ip.address.of.tpu:8470 "
        "url.")

    flags.DEFINE_string(
        "tpu_zone", None,
        "[Optional] GCE zone where the Cloud TPU is located in. If not "
        "specified, we will attempt to automatically detect the GCE project from "
        "metadata.")

    flags.DEFINE_string(
        "gcp_project", None,
        "[Optional] Project name for the Cloud TPU-enabled project. If not "
        "specified, we will attempt to automatically detect the GCE project from "
        "metadata.")

    flags.DEFINE_string("master", None, "[Optional] TensorFlow master URL.")

    flags.DEFINE_integer(
        "num_tpu_cores", 8,
        "Only used if `use_tpu` is True. Total number of TPU cores to use.")

    flags.DEFINE_bool(
        "verbose_logging", False,
        "If true, all of the warnings related to data processing will be printed. "
        "A number of warnings are expected for a normal NQ evaluation.")

    flags.DEFINE_boolean(
        "skip_nested_contexts", True,
        "Completely ignore context that are not top level nodes in the page.")

    flags.DEFINE_integer("task_id", 0,
                         "Train and dev shard to read from and write to.")

    flags.DEFINE_integer("max_contexts", 48,
                         "Maximum number of contexts to output for an example.")

    flags.DEFINE_integer(
        "max_position", 50,
        "Maximum context position for which to generate special tokens.")

