import paddle.fluid as fluid
import joblib
from model.ernie import ErnieConfig
#from finetune.classifier import create_model
from train import create_model
import numpy as np
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--init_pretraining_params", default='./ERNIE_Raw/params', type=str, help=".")
parser.add_argument("--ernie_config_path", default='./pytorch/bert_config.json', type=str, help=".")
parser.add_argument("--max_seq_len", default=10, type=int, help=".")
parser.add_argument("--num_labels", default=2, type=int, help=".")
parser.add_argument("--use_fp16", type=bool, default=False, help="Whether to use fp16 mixed precision training.")
parser.add_argument("--loss_scaling",  type=float,  default=1.0, help="only valid when use_fp16 is enabled.")
startup_prog = fluid.Program()
test_prog = fluid.Program()

place = fluid.CPUPlace()
exe = fluid.Executor(place)
exe.run(startup_prog)

args = parser.parse_args()

ernie_config = ErnieConfig(args.ernie_config_path)
ernie_config.print_config()

with fluid.program_guard(test_prog, startup_prog):
    with fluid.unique_name.guard():
        #train_pyreader, graph_vars = create_model(args, pyreader_name='train_reader', ernie_config=ernie_config)
        train_pyreader, next_sent_acc, mask_lm_loss, total_loss = create_model(pyreader_name='train_reader', ernie_config=ernie_config)
        
def if_exist(var):
    #print(var.name)
    return os.path.exists(os.path.join(args.init_pretraining_params,var.name))

fluid.io.load_vars(exe, args.init_pretraining_params, main_program=test_prog, predicate=if_exist)
var_dict = {}
for var in startup_prog.list_vars():
    if os.path.exists(os.path.join(args.init_pretraining_params,var.name)):
    #if ( 'encoder' in var.name or 'embedding' in var.name ) and 'tmp' not in var.name:
    #if 'tmp' not in var.name:
        fluid_tensor = fluid.global_scope().find_var(var.name).get_tensor()
        print(var.name, np.array(fluid_tensor).shape)
        var_dict[var.name] = np.array(fluid_tensor)

joblib.dump(var_dict, './paddlepaddle/model.dct')