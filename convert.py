import torch
import joblib
import numpy as np
from collections import OrderedDict

from BERT import BertConfig, BertForMaskedLM

var_dict = joblib.load("./paddlepaddle/model.dct")

'''
for key in var_dict.keys():
    print("key: ", key)
    print("val: ", np.array(var_dict[key]).shape)
'''
# (18000, 768)
word_embedding_array = np.array(var_dict["word_embedding"])
# (513, 768)
pos_embedding_array = np.array(var_dict["pos_embedding"])
# (2, 768)
sent_embedding_array = np.array(var_dict["sent_embedding"])

# (768)
pre_encoder_layer_norm_scale_array = np.array(var_dict["pre_encoder_layer_norm_scale"])
# (768)
pre_encoder_layer_norm_bias_array = np.array(var_dict["pre_encoder_layer_norm_bias"])

# (768, 768)
encoder_layer_0_multi_head_att_query_fc_w_0_array = np.array(var_dict["encoder_layer_0_multi_head_att_query_fc.w_0"])
# (768)
encoder_layer_0_multi_head_att_query_fc_b_0_array = np.array(var_dict["encoder_layer_0_multi_head_att_query_fc.b_0"])
# (768, 768)
encoder_layer_0_multi_head_att_key_fc_w_0_array = np.array(var_dict["encoder_layer_0_multi_head_att_key_fc.w_0"])
# (768)
encoder_layer_0_multi_head_att_key_fc_b_0_array = np.array(var_dict["encoder_layer_0_multi_head_att_key_fc.b_0"])
# (768, 768)
encoder_layer_0_multi_head_att_value_fc_w_0_array = np.array(var_dict["encoder_layer_0_multi_head_att_value_fc.w_0"])
# (768)
encoder_layer_0_multi_head_att_value_fc_b_0_array = np.array(var_dict["encoder_layer_0_multi_head_att_value_fc.b_0"])
# (768, 768)
encoder_layer_0_multi_head_att_output_fc_w_0_array = np.array(var_dict["encoder_layer_0_multi_head_att_output_fc.w_0"])
# (768)
encoder_layer_0_multi_head_att_output_fc_b_0_array = np.array(var_dict["encoder_layer_0_multi_head_att_output_fc.b_0"])
# (768)
encoder_layer_0_post_att_layer_norm_scale_array = np.array(var_dict["encoder_layer_0_post_att_layer_norm_scale"])
# (768)
encoder_layer_0_post_att_layer_norm_bias_array = np.array(var_dict["encoder_layer_0_post_att_layer_norm_bias"])
# (768, 3072)
encoder_layer_0_ffn_fc_0_w_0_array = np.array(var_dict["encoder_layer_0_ffn_fc_0.w_0"])
# (3072)
encoder_layer_0_ffn_fc_0_b_0_array = np.array(var_dict["encoder_layer_0_ffn_fc_0.b_0"])
# (3072, 768)
encoder_layer_0_ffn_fc_1_w_0_array = np.array(var_dict["encoder_layer_0_ffn_fc_1.w_0"])
# (768)
encoder_layer_0_ffn_fc_1_b_0_array = np.array(var_dict["encoder_layer_0_ffn_fc_1.b_0"])
# (768)
encoder_layer_0_post_ffn_layer_norm_scale_array = np.array(var_dict["encoder_layer_0_post_ffn_layer_norm_scale"])
# (768)
encoder_layer_0_post_ffn_layer_norm_bias_array = np.array(var_dict["encoder_layer_0_post_ffn_layer_norm_bias"])

# (768, 768)
encoder_layer_1_multi_head_att_query_fc_w_0_array = np.array(var_dict["encoder_layer_1_multi_head_att_query_fc.w_0"])
# (768)
encoder_layer_1_multi_head_att_query_fc_b_0_array = np.array(var_dict["encoder_layer_1_multi_head_att_query_fc.b_0"])
# (768, 768)
encoder_layer_1_multi_head_att_key_fc_w_0_array = np.array(var_dict["encoder_layer_1_multi_head_att_key_fc.w_0"])
# (768)
encoder_layer_1_multi_head_att_key_fc_b_0_array = np.array(var_dict["encoder_layer_1_multi_head_att_key_fc.b_0"])
# (768, 768)
encoder_layer_1_multi_head_att_value_fc_w_0_array = np.array(var_dict["encoder_layer_1_multi_head_att_value_fc.w_0"])
# (768)
encoder_layer_1_multi_head_att_value_fc_b_0_array = np.array(var_dict["encoder_layer_1_multi_head_att_value_fc.b_0"])
# (768, 768)
encoder_layer_1_multi_head_att_output_fc_w_0_array = np.array(var_dict["encoder_layer_1_multi_head_att_output_fc.w_0"])
# (768)
encoder_layer_1_multi_head_att_output_fc_b_0_array = np.array(var_dict["encoder_layer_1_multi_head_att_output_fc.b_0"])
# (768)
encoder_layer_1_post_att_layer_norm_scale_array = np.array(var_dict["encoder_layer_1_post_att_layer_norm_scale"])
# (768)
encoder_layer_1_post_att_layer_norm_bias_array = np.array(var_dict["encoder_layer_1_post_att_layer_norm_bias"])
# (768, 3072)
encoder_layer_1_ffn_fc_0_w_0_array = np.array(var_dict["encoder_layer_1_ffn_fc_0.w_0"])
# (3072)
encoder_layer_1_ffn_fc_0_b_0_array = np.array(var_dict["encoder_layer_1_ffn_fc_0.b_0"])
# (3072, 768)
encoder_layer_1_ffn_fc_1_w_0_array = np.array(var_dict["encoder_layer_1_ffn_fc_1.w_0"])
# (768)
encoder_layer_1_ffn_fc_1_b_0_array = np.array(var_dict["encoder_layer_1_ffn_fc_1.b_0"])
# (768)
encoder_layer_1_post_ffn_layer_norm_scale_array = np.array(var_dict["encoder_layer_1_post_ffn_layer_norm_scale"])
# (768)
encoder_layer_1_post_ffn_layer_norm_bias_array = np.array(var_dict["encoder_layer_1_post_ffn_layer_norm_bias"])

# (768, 768)
encoder_layer_2_multi_head_att_query_fc_w_0_array = np.array(var_dict["encoder_layer_2_multi_head_att_query_fc.w_0"])
# (768)
encoder_layer_2_multi_head_att_query_fc_b_0_array = np.array(var_dict["encoder_layer_2_multi_head_att_query_fc.b_0"])
# (768, 768)
encoder_layer_2_multi_head_att_key_fc_w_0_array = np.array(var_dict["encoder_layer_2_multi_head_att_key_fc.w_0"])
# (768)
encoder_layer_2_multi_head_att_key_fc_b_0_array = np.array(var_dict["encoder_layer_2_multi_head_att_key_fc.b_0"])
# (768, 768)
encoder_layer_2_multi_head_att_value_fc_w_0_array = np.array(var_dict["encoder_layer_2_multi_head_att_value_fc.w_0"])
# (768)
encoder_layer_2_multi_head_att_value_fc_b_0_array = np.array(var_dict["encoder_layer_2_multi_head_att_value_fc.b_0"])
# (768, 768)
encoder_layer_2_multi_head_att_output_fc_w_0_array = np.array(var_dict["encoder_layer_2_multi_head_att_output_fc.w_0"])
# (768)
encoder_layer_2_multi_head_att_output_fc_b_0_array = np.array(var_dict["encoder_layer_2_multi_head_att_output_fc.b_0"])
# (768)
encoder_layer_2_post_att_layer_norm_scale_array = np.array(var_dict["encoder_layer_2_post_att_layer_norm_scale"])
# (768)
encoder_layer_2_post_att_layer_norm_bias_array = np.array(var_dict["encoder_layer_2_post_att_layer_norm_bias"])
# (768, 3072)
encoder_layer_2_ffn_fc_0_w_0_array = np.array(var_dict["encoder_layer_2_ffn_fc_0.w_0"])
# (3072)
encoder_layer_2_ffn_fc_0_b_0_array = np.array(var_dict["encoder_layer_2_ffn_fc_0.b_0"])
# (3072, 768)
encoder_layer_2_ffn_fc_1_w_0_array = np.array(var_dict["encoder_layer_2_ffn_fc_1.w_0"])
# (768)
encoder_layer_2_ffn_fc_1_b_0_array = np.array(var_dict["encoder_layer_2_ffn_fc_1.b_0"])
# (768)
encoder_layer_2_post_ffn_layer_norm_scale_array = np.array(var_dict["encoder_layer_2_post_ffn_layer_norm_scale"])
# (768)
encoder_layer_2_post_ffn_layer_norm_bias_array = np.array(var_dict["encoder_layer_2_post_ffn_layer_norm_bias"])

# (768, 768)
encoder_layer_3_multi_head_att_query_fc_w_0_array = np.array(var_dict["encoder_layer_3_multi_head_att_query_fc.w_0"])
# (768)
encoder_layer_3_multi_head_att_query_fc_b_0_array = np.array(var_dict["encoder_layer_3_multi_head_att_query_fc.b_0"])
# (768, 768)
encoder_layer_3_multi_head_att_key_fc_w_0_array = np.array(var_dict["encoder_layer_3_multi_head_att_key_fc.w_0"])
# (768)
encoder_layer_3_multi_head_att_key_fc_b_0_array = np.array(var_dict["encoder_layer_3_multi_head_att_key_fc.b_0"])
# (768, 768)
encoder_layer_3_multi_head_att_value_fc_w_0_array = np.array(var_dict["encoder_layer_3_multi_head_att_value_fc.w_0"])
# (768)
encoder_layer_3_multi_head_att_value_fc_b_0_array = np.array(var_dict["encoder_layer_3_multi_head_att_value_fc.b_0"])
# (768, 768)
encoder_layer_3_multi_head_att_output_fc_w_0_array = np.array(var_dict["encoder_layer_3_multi_head_att_output_fc.w_0"])
# (768)
encoder_layer_3_multi_head_att_output_fc_b_0_array = np.array(var_dict["encoder_layer_3_multi_head_att_output_fc.b_0"])
# (768)
encoder_layer_3_post_att_layer_norm_scale_array = np.array(var_dict["encoder_layer_3_post_att_layer_norm_scale"])
# (768)
encoder_layer_3_post_att_layer_norm_bias_array = np.array(var_dict["encoder_layer_3_post_att_layer_norm_bias"])
# (768, 3072)
encoder_layer_3_ffn_fc_0_w_0_array = np.array(var_dict["encoder_layer_3_ffn_fc_0.w_0"])
# (3072)
encoder_layer_3_ffn_fc_0_b_0_array = np.array(var_dict["encoder_layer_3_ffn_fc_0.b_0"])
# (3072, 768)
encoder_layer_3_ffn_fc_1_w_0_array = np.array(var_dict["encoder_layer_3_ffn_fc_1.w_0"])
# (768)
encoder_layer_3_ffn_fc_1_b_0_array = np.array(var_dict["encoder_layer_3_ffn_fc_1.b_0"])
# (768)
encoder_layer_3_post_ffn_layer_norm_scale_array = np.array(var_dict["encoder_layer_3_post_ffn_layer_norm_scale"])
# (768)
encoder_layer_3_post_ffn_layer_norm_bias_array = np.array(var_dict["encoder_layer_3_post_ffn_layer_norm_bias"])

# (768, 768)
encoder_layer_4_multi_head_att_query_fc_w_0_array = np.array(var_dict["encoder_layer_4_multi_head_att_query_fc.w_0"])
# (768)
encoder_layer_4_multi_head_att_query_fc_b_0_array = np.array(var_dict["encoder_layer_4_multi_head_att_query_fc.b_0"])
# (768, 768)
encoder_layer_4_multi_head_att_key_fc_w_0_array = np.array(var_dict["encoder_layer_4_multi_head_att_key_fc.w_0"])
# (768)
encoder_layer_4_multi_head_att_key_fc_b_0_array = np.array(var_dict["encoder_layer_4_multi_head_att_key_fc.b_0"])
# (768, 768)
encoder_layer_4_multi_head_att_value_fc_w_0_array = np.array(var_dict["encoder_layer_4_multi_head_att_value_fc.w_0"])
# (768)
encoder_layer_4_multi_head_att_value_fc_b_0_array = np.array(var_dict["encoder_layer_4_multi_head_att_value_fc.b_0"])
# (768, 768)
encoder_layer_4_multi_head_att_output_fc_w_0_array = np.array(var_dict["encoder_layer_4_multi_head_att_output_fc.w_0"])
# (768)
encoder_layer_4_multi_head_att_output_fc_b_0_array = np.array(var_dict["encoder_layer_4_multi_head_att_output_fc.b_0"])
# (768)
encoder_layer_4_post_att_layer_norm_scale_array = np.array(var_dict["encoder_layer_4_post_att_layer_norm_scale"])
# (768)
encoder_layer_4_post_att_layer_norm_bias_array = np.array(var_dict["encoder_layer_4_post_att_layer_norm_bias"])
# (768, 3072)
encoder_layer_4_ffn_fc_0_w_0_array = np.array(var_dict["encoder_layer_4_ffn_fc_0.w_0"])
# (3072)
encoder_layer_4_ffn_fc_0_b_0_array = np.array(var_dict["encoder_layer_4_ffn_fc_0.b_0"])
# (3072, 768)
encoder_layer_4_ffn_fc_1_w_0_array = np.array(var_dict["encoder_layer_4_ffn_fc_1.w_0"])
# (768)
encoder_layer_4_ffn_fc_1_b_0_array = np.array(var_dict["encoder_layer_4_ffn_fc_1.b_0"])
# (768)
encoder_layer_4_post_ffn_layer_norm_scale_array = np.array(var_dict["encoder_layer_4_post_ffn_layer_norm_scale"])
# (768)
encoder_layer_4_post_ffn_layer_norm_bias_array = np.array(var_dict["encoder_layer_4_post_ffn_layer_norm_bias"])

# (768, 768)
encoder_layer_5_multi_head_att_query_fc_w_0_array = np.array(var_dict["encoder_layer_5_multi_head_att_query_fc.w_0"])
# (768)
encoder_layer_5_multi_head_att_query_fc_b_0_array = np.array(var_dict["encoder_layer_5_multi_head_att_query_fc.b_0"])
# (768, 768)
encoder_layer_5_multi_head_att_key_fc_w_0_array = np.array(var_dict["encoder_layer_5_multi_head_att_key_fc.w_0"])
# (768)
encoder_layer_5_multi_head_att_key_fc_b_0_array = np.array(var_dict["encoder_layer_5_multi_head_att_key_fc.b_0"])
# (768, 768)
encoder_layer_5_multi_head_att_value_fc_w_0_array = np.array(var_dict["encoder_layer_5_multi_head_att_value_fc.w_0"])
# (768)
encoder_layer_5_multi_head_att_value_fc_b_0_array = np.array(var_dict["encoder_layer_5_multi_head_att_value_fc.b_0"])
# (768, 768)
encoder_layer_5_multi_head_att_output_fc_w_0_array = np.array(var_dict["encoder_layer_5_multi_head_att_output_fc.w_0"])
# (768)
encoder_layer_5_multi_head_att_output_fc_b_0_array = np.array(var_dict["encoder_layer_5_multi_head_att_output_fc.b_0"])
# (768)
encoder_layer_5_post_att_layer_norm_scale_array = np.array(var_dict["encoder_layer_5_post_att_layer_norm_scale"])
# (768)
encoder_layer_5_post_att_layer_norm_bias_array = np.array(var_dict["encoder_layer_5_post_att_layer_norm_bias"])
# (768, 3072)
encoder_layer_5_ffn_fc_0_w_0_array = np.array(var_dict["encoder_layer_5_ffn_fc_0.w_0"])
# (3072)
encoder_layer_5_ffn_fc_0_b_0_array = np.array(var_dict["encoder_layer_5_ffn_fc_0.b_0"])
# (3072, 768)
encoder_layer_5_ffn_fc_1_w_0_array = np.array(var_dict["encoder_layer_5_ffn_fc_1.w_0"])
# (768)
encoder_layer_5_ffn_fc_1_b_0_array = np.array(var_dict["encoder_layer_5_ffn_fc_1.b_0"])
# (768)
encoder_layer_5_post_ffn_layer_norm_scale_array = np.array(var_dict["encoder_layer_5_post_ffn_layer_norm_scale"])
# (768)
encoder_layer_5_post_ffn_layer_norm_bias_array = np.array(var_dict["encoder_layer_5_post_ffn_layer_norm_bias"])

# (768, 768)
encoder_layer_6_multi_head_att_query_fc_w_0_array = np.array(var_dict["encoder_layer_6_multi_head_att_query_fc.w_0"])
# (768)
encoder_layer_6_multi_head_att_query_fc_b_0_array = np.array(var_dict["encoder_layer_6_multi_head_att_query_fc.b_0"])
# (768, 768)
encoder_layer_6_multi_head_att_key_fc_w_0_array = np.array(var_dict["encoder_layer_6_multi_head_att_key_fc.w_0"])
# (768)
encoder_layer_6_multi_head_att_key_fc_b_0_array = np.array(var_dict["encoder_layer_6_multi_head_att_key_fc.b_0"])
# (768, 768)
encoder_layer_6_multi_head_att_value_fc_w_0_array = np.array(var_dict["encoder_layer_6_multi_head_att_value_fc.w_0"])
# (768)
encoder_layer_6_multi_head_att_value_fc_b_0_array = np.array(var_dict["encoder_layer_6_multi_head_att_value_fc.b_0"])
# (768, 768)
encoder_layer_6_multi_head_att_output_fc_w_0_array = np.array(var_dict["encoder_layer_6_multi_head_att_output_fc.w_0"])
# (768)
encoder_layer_6_multi_head_att_output_fc_b_0_array = np.array(var_dict["encoder_layer_6_multi_head_att_output_fc.b_0"])
# (768)
encoder_layer_6_post_att_layer_norm_scale_array = np.array(var_dict["encoder_layer_6_post_att_layer_norm_scale"])
# (768)
encoder_layer_6_post_att_layer_norm_bias_array = np.array(var_dict["encoder_layer_6_post_att_layer_norm_bias"])
# (768, 3072)
encoder_layer_6_ffn_fc_0_w_0_array = np.array(var_dict["encoder_layer_6_ffn_fc_0.w_0"])
# (3072)
encoder_layer_6_ffn_fc_0_b_0_array = np.array(var_dict["encoder_layer_6_ffn_fc_0.b_0"])
# (3072, 768)
encoder_layer_6_ffn_fc_1_w_0_array = np.array(var_dict["encoder_layer_6_ffn_fc_1.w_0"])
# (768)
encoder_layer_6_ffn_fc_1_b_0_array = np.array(var_dict["encoder_layer_6_ffn_fc_1.b_0"])
# (768)
encoder_layer_6_post_ffn_layer_norm_scale_array = np.array(var_dict["encoder_layer_6_post_ffn_layer_norm_scale"])
# (768)
encoder_layer_6_post_ffn_layer_norm_bias_array = np.array(var_dict["encoder_layer_6_post_ffn_layer_norm_bias"])

# (768, 768)
encoder_layer_7_multi_head_att_query_fc_w_0_array = np.array(var_dict["encoder_layer_7_multi_head_att_query_fc.w_0"])
# (768)
encoder_layer_7_multi_head_att_query_fc_b_0_array = np.array(var_dict["encoder_layer_7_multi_head_att_query_fc.b_0"])
# (768, 768)
encoder_layer_7_multi_head_att_key_fc_w_0_array = np.array(var_dict["encoder_layer_7_multi_head_att_key_fc.w_0"])
# (768)
encoder_layer_7_multi_head_att_key_fc_b_0_array = np.array(var_dict["encoder_layer_7_multi_head_att_key_fc.b_0"])
# (768, 768)
encoder_layer_7_multi_head_att_value_fc_w_0_array = np.array(var_dict["encoder_layer_7_multi_head_att_value_fc.w_0"])
# (768)
encoder_layer_7_multi_head_att_value_fc_b_0_array = np.array(var_dict["encoder_layer_7_multi_head_att_value_fc.b_0"])
# (768, 768)
encoder_layer_7_multi_head_att_output_fc_w_0_array = np.array(var_dict["encoder_layer_7_multi_head_att_output_fc.w_0"])
# (768)
encoder_layer_7_multi_head_att_output_fc_b_0_array = np.array(var_dict["encoder_layer_7_multi_head_att_output_fc.b_0"])
# (768)
encoder_layer_7_post_att_layer_norm_scale_array = np.array(var_dict["encoder_layer_7_post_att_layer_norm_scale"])
# (768)
encoder_layer_7_post_att_layer_norm_bias_array = np.array(var_dict["encoder_layer_7_post_att_layer_norm_bias"])
# (768, 3072)
encoder_layer_7_ffn_fc_0_w_0_array = np.array(var_dict["encoder_layer_7_ffn_fc_0.w_0"])
# (3072)
encoder_layer_7_ffn_fc_0_b_0_array = np.array(var_dict["encoder_layer_7_ffn_fc_0.b_0"])
# (3072, 768)
encoder_layer_7_ffn_fc_1_w_0_array = np.array(var_dict["encoder_layer_7_ffn_fc_1.w_0"])
# (768)
encoder_layer_7_ffn_fc_1_b_0_array = np.array(var_dict["encoder_layer_7_ffn_fc_1.b_0"])
# (768)
encoder_layer_7_post_ffn_layer_norm_scale_array = np.array(var_dict["encoder_layer_7_post_ffn_layer_norm_scale"])
# (768)
encoder_layer_7_post_ffn_layer_norm_bias_array = np.array(var_dict["encoder_layer_7_post_ffn_layer_norm_bias"])

# (768, 768)
encoder_layer_8_multi_head_att_query_fc_w_0_array = np.array(var_dict["encoder_layer_8_multi_head_att_query_fc.w_0"])
# (768)
encoder_layer_8_multi_head_att_query_fc_b_0_array = np.array(var_dict["encoder_layer_8_multi_head_att_query_fc.b_0"])
# (768, 768)
encoder_layer_8_multi_head_att_key_fc_w_0_array = np.array(var_dict["encoder_layer_8_multi_head_att_key_fc.w_0"])
# (768)
encoder_layer_8_multi_head_att_key_fc_b_0_array = np.array(var_dict["encoder_layer_8_multi_head_att_key_fc.b_0"])
# (768, 768)
encoder_layer_8_multi_head_att_value_fc_w_0_array = np.array(var_dict["encoder_layer_8_multi_head_att_value_fc.w_0"])
# (768)
encoder_layer_8_multi_head_att_value_fc_b_0_array = np.array(var_dict["encoder_layer_8_multi_head_att_value_fc.b_0"])
# (768, 768)
encoder_layer_8_multi_head_att_output_fc_w_0_array = np.array(var_dict["encoder_layer_8_multi_head_att_output_fc.w_0"])
# (768)
encoder_layer_8_multi_head_att_output_fc_b_0_array = np.array(var_dict["encoder_layer_8_multi_head_att_output_fc.b_0"])
# (768)
encoder_layer_8_post_att_layer_norm_scale_array = np.array(var_dict["encoder_layer_8_post_att_layer_norm_scale"])
# (768)
encoder_layer_8_post_att_layer_norm_bias_array = np.array(var_dict["encoder_layer_8_post_att_layer_norm_bias"])
# (768, 3072)
encoder_layer_8_ffn_fc_0_w_0_array = np.array(var_dict["encoder_layer_8_ffn_fc_0.w_0"])
# (3072)
encoder_layer_8_ffn_fc_0_b_0_array = np.array(var_dict["encoder_layer_8_ffn_fc_0.b_0"])
# (3072, 768)
encoder_layer_8_ffn_fc_1_w_0_array = np.array(var_dict["encoder_layer_8_ffn_fc_1.w_0"])
# (768)
encoder_layer_8_ffn_fc_1_b_0_array = np.array(var_dict["encoder_layer_8_ffn_fc_1.b_0"])
# (768)
encoder_layer_8_post_ffn_layer_norm_scale_array = np.array(var_dict["encoder_layer_8_post_ffn_layer_norm_scale"])
# (768)
encoder_layer_8_post_ffn_layer_norm_bias_array = np.array(var_dict["encoder_layer_8_post_ffn_layer_norm_bias"])

# (768, 768)
encoder_layer_9_multi_head_att_query_fc_w_0_array = np.array(var_dict["encoder_layer_9_multi_head_att_query_fc.w_0"])
# (768)
encoder_layer_9_multi_head_att_query_fc_b_0_array = np.array(var_dict["encoder_layer_9_multi_head_att_query_fc.b_0"])
# (768, 768)
encoder_layer_9_multi_head_att_key_fc_w_0_array = np.array(var_dict["encoder_layer_9_multi_head_att_key_fc.w_0"])
# (768)
encoder_layer_9_multi_head_att_key_fc_b_0_array = np.array(var_dict["encoder_layer_9_multi_head_att_key_fc.b_0"])
# (768, 768)
encoder_layer_9_multi_head_att_value_fc_w_0_array = np.array(var_dict["encoder_layer_9_multi_head_att_value_fc.w_0"])
# (768)
encoder_layer_9_multi_head_att_value_fc_b_0_array = np.array(var_dict["encoder_layer_9_multi_head_att_value_fc.b_0"])
# (768, 768)
encoder_layer_9_multi_head_att_output_fc_w_0_array = np.array(var_dict["encoder_layer_9_multi_head_att_output_fc.w_0"])
# (768)
encoder_layer_9_multi_head_att_output_fc_b_0_array = np.array(var_dict["encoder_layer_9_multi_head_att_output_fc.b_0"])
# (768)
encoder_layer_9_post_att_layer_norm_scale_array = np.array(var_dict["encoder_layer_9_post_att_layer_norm_scale"])
# (768)
encoder_layer_9_post_att_layer_norm_bias_array = np.array(var_dict["encoder_layer_9_post_att_layer_norm_bias"])
# (768, 3072)
encoder_layer_9_ffn_fc_0_w_0_array = np.array(var_dict["encoder_layer_9_ffn_fc_0.w_0"])
# (3072)
encoder_layer_9_ffn_fc_0_b_0_array = np.array(var_dict["encoder_layer_9_ffn_fc_0.b_0"])
# (3072, 768)
encoder_layer_9_ffn_fc_1_w_0_array = np.array(var_dict["encoder_layer_9_ffn_fc_1.w_0"])
# (768)
encoder_layer_9_ffn_fc_1_b_0_array = np.array(var_dict["encoder_layer_9_ffn_fc_1.b_0"])
# (768)
encoder_layer_9_post_ffn_layer_norm_scale_array = np.array(var_dict["encoder_layer_9_post_ffn_layer_norm_scale"])
# (768)
encoder_layer_9_post_ffn_layer_norm_bias_array = np.array(var_dict["encoder_layer_9_post_ffn_layer_norm_bias"])

# (768, 768)
encoder_layer_10_multi_head_att_query_fc_w_0_array = np.array(var_dict["encoder_layer_10_multi_head_att_query_fc.w_0"])
# (768)
encoder_layer_10_multi_head_att_query_fc_b_0_array = np.array(var_dict["encoder_layer_10_multi_head_att_query_fc.b_0"])
# (768, 768)
encoder_layer_10_multi_head_att_key_fc_w_0_array = np.array(var_dict["encoder_layer_10_multi_head_att_key_fc.w_0"])
# (768)
encoder_layer_10_multi_head_att_key_fc_b_0_array = np.array(var_dict["encoder_layer_10_multi_head_att_key_fc.b_0"])
# (768, 768)
encoder_layer_10_multi_head_att_value_fc_w_0_array = np.array(var_dict["encoder_layer_10_multi_head_att_value_fc.w_0"])
# (768)
encoder_layer_10_multi_head_att_value_fc_b_0_array = np.array(var_dict["encoder_layer_10_multi_head_att_value_fc.b_0"])
# (768, 768)
encoder_layer_10_multi_head_att_output_fc_w_0_array = np.array(var_dict["encoder_layer_10_multi_head_att_output_fc.w_0"])
# (768)
encoder_layer_10_multi_head_att_output_fc_b_0_array = np.array(var_dict["encoder_layer_10_multi_head_att_output_fc.b_0"])
# (768)
encoder_layer_10_post_att_layer_norm_scale_array = np.array(var_dict["encoder_layer_10_post_att_layer_norm_scale"])
# (768)
encoder_layer_10_post_att_layer_norm_bias_array = np.array(var_dict["encoder_layer_10_post_att_layer_norm_bias"])
# (768, 3072)
encoder_layer_10_ffn_fc_0_w_0_array = np.array(var_dict["encoder_layer_10_ffn_fc_0.w_0"])
# (3072)
encoder_layer_10_ffn_fc_0_b_0_array = np.array(var_dict["encoder_layer_10_ffn_fc_0.b_0"])
# (3072, 768)
encoder_layer_10_ffn_fc_1_w_0_array = np.array(var_dict["encoder_layer_10_ffn_fc_1.w_0"])
# (768)
encoder_layer_10_ffn_fc_1_b_0_array = np.array(var_dict["encoder_layer_10_ffn_fc_1.b_0"])
# (768)
encoder_layer_10_post_ffn_layer_norm_scale_array = np.array(var_dict["encoder_layer_10_post_ffn_layer_norm_scale"])
# (768)
encoder_layer_10_post_ffn_layer_norm_bias_array = np.array(var_dict["encoder_layer_10_post_ffn_layer_norm_bias"])

# (768, 768)
encoder_layer_11_multi_head_att_query_fc_w_0_array = np.array(var_dict["encoder_layer_11_multi_head_att_query_fc.w_0"])
# (768)
encoder_layer_11_multi_head_att_query_fc_b_0_array = np.array(var_dict["encoder_layer_11_multi_head_att_query_fc.b_0"])
# (768, 768)
encoder_layer_11_multi_head_att_key_fc_w_0_array = np.array(var_dict["encoder_layer_11_multi_head_att_key_fc.w_0"])
# (768)
encoder_layer_11_multi_head_att_key_fc_b_0_array = np.array(var_dict["encoder_layer_11_multi_head_att_key_fc.b_0"])
# (768, 768)
encoder_layer_11_multi_head_att_value_fc_w_0_array = np.array(var_dict["encoder_layer_11_multi_head_att_value_fc.w_0"])
# (768)
encoder_layer_11_multi_head_att_value_fc_b_0_array = np.array(var_dict["encoder_layer_11_multi_head_att_value_fc.b_0"])
# (768, 768)
encoder_layer_11_multi_head_att_output_fc_w_0_array = np.array(var_dict["encoder_layer_11_multi_head_att_output_fc.w_0"])
# (768)
encoder_layer_11_multi_head_att_output_fc_b_0_array = np.array(var_dict["encoder_layer_11_multi_head_att_output_fc.b_0"])
# (768)
encoder_layer_11_post_att_layer_norm_scale_array = np.array(var_dict["encoder_layer_11_post_att_layer_norm_scale"])
# (768)
encoder_layer_11_post_att_layer_norm_bias_array = np.array(var_dict["encoder_layer_11_post_att_layer_norm_bias"])
# (768, 3072)
encoder_layer_11_ffn_fc_0_w_0_array = np.array(var_dict["encoder_layer_11_ffn_fc_0.w_0"])
# (3072)
encoder_layer_11_ffn_fc_0_b_0_array = np.array(var_dict["encoder_layer_11_ffn_fc_0.b_0"])
# (3072, 768)
encoder_layer_11_ffn_fc_1_w_0_array = np.array(var_dict["encoder_layer_11_ffn_fc_1.w_0"])
# (768)
encoder_layer_11_ffn_fc_1_b_0_array = np.array(var_dict["encoder_layer_11_ffn_fc_1.b_0"])
# (768)
encoder_layer_11_post_ffn_layer_norm_scale_array = np.array(var_dict["encoder_layer_11_post_ffn_layer_norm_scale"])
# (768)
encoder_layer_11_post_ffn_layer_norm_bias_array = np.array(var_dict["encoder_layer_11_post_ffn_layer_norm_bias"])

# (768, 768)
pooled_fc_w_0_array = np.array(var_dict["pooled_fc.w_0"])
# (768)
pooled_fc_b_0_array = np.array(var_dict["pooled_fc.b_0"])

# (768, 768)
mask_lm_trans_fc_w_0_array = np.array(var_dict["mask_lm_trans_fc.w_0"])
# (768)
mask_lm_trans_fc_b_0_array = np.array(var_dict["mask_lm_trans_fc.b_0"])
# (768)
mask_lm_trans_layer_norm_scale_array = np.array(var_dict["mask_lm_trans_layer_norm_scale"])
# (768)
mask_lm_trans_layer_norm_bias_array = np.array(var_dict["mask_lm_trans_layer_norm_bias"])
# (18000, )
mask_lm_out_fc_b_0_array = np.array(var_dict["mask_lm_out_fc.b_0"])

# Initilaize pytorch model
config = BertConfig.from_json_file("./pytorch/bert_config.json")
print("Building pytorch model from configuration: {}".format(str(config)))
model = BertForMaskedLM(config)

'''
print("pytorch model: ")
for name, param in model.named_parameters():
    print(name, param.size())
'''

for k, v in model.state_dict().items():
    print(k, v.size())

#print(model.state_dict()["cls.predictions.transform.LayerNorm.bias"])

new_state_dict = OrderedDict({"bert.embeddings.word_embeddings.weight": torch.from_numpy(word_embedding_array),
                                "bert.embeddings.position_embeddings.weight": torch.from_numpy(pos_embedding_array), 
                                "bert.embeddings.token_type_embeddings.weight": torch.from_numpy(sent_embedding_array),
                                "bert.embeddings.LayerNorm.weight": torch.from_numpy(pre_encoder_layer_norm_scale_array),
                                "bert.embeddings.LayerNorm.bias": torch.from_numpy(pre_encoder_layer_norm_bias_array),  

                                "bert.encoder.layer.0.attention.self.query.weight": torch.from_numpy(np.transpose(encoder_layer_0_multi_head_att_query_fc_w_0_array)),
                                "bert.encoder.layer.0.attention.self.query.bias": torch.from_numpy(encoder_layer_0_multi_head_att_query_fc_b_0_array),
                                "bert.encoder.layer.0.attention.self.key.weight": torch.from_numpy(np.transpose(encoder_layer_0_multi_head_att_key_fc_w_0_array)),
                                "bert.encoder.layer.0.attention.self.key.bias": torch.from_numpy(encoder_layer_0_multi_head_att_key_fc_b_0_array),
                                "bert.encoder.layer.0.attention.self.value.weight": torch.from_numpy(np.transpose(encoder_layer_0_multi_head_att_value_fc_w_0_array)),
                                "bert.encoder.layer.0.attention.self.value.bias": torch.from_numpy(encoder_layer_0_multi_head_att_value_fc_b_0_array),
                                "bert.encoder.layer.0.attention.output.dense.weight": torch.from_numpy(np.transpose(encoder_layer_0_multi_head_att_output_fc_w_0_array)),
                                "bert.encoder.layer.0.attention.output.dense.bias": torch.from_numpy(encoder_layer_0_multi_head_att_output_fc_b_0_array),
                                "bert.encoder.layer.0.attention.output.LayerNorm.weight": torch.from_numpy(encoder_layer_0_post_att_layer_norm_scale_array),
                                "bert.encoder.layer.0.attention.output.LayerNorm.bias": torch.from_numpy(encoder_layer_0_post_att_layer_norm_bias_array),
                                "bert.encoder.layer.0.intermediate.dense.weight": torch.from_numpy(np.transpose(encoder_layer_0_ffn_fc_0_w_0_array)),
                                "bert.encoder.layer.0.intermediate.dense.bias": torch.from_numpy(encoder_layer_0_ffn_fc_0_b_0_array),
                                "bert.encoder.layer.0.output.dense.weight": torch.from_numpy(np.transpose(encoder_layer_0_ffn_fc_1_w_0_array)),
                                "bert.encoder.layer.0.output.dense.bias": torch.from_numpy(encoder_layer_0_ffn_fc_1_b_0_array),
                                "bert.encoder.layer.0.output.LayerNorm.weight": torch.from_numpy(encoder_layer_0_post_ffn_layer_norm_scale_array),
                                "bert.encoder.layer.0.output.LayerNorm.bias": torch.from_numpy(encoder_layer_0_post_ffn_layer_norm_bias_array),

                                "bert.encoder.layer.1.attention.self.query.weight": torch.from_numpy(np.transpose(encoder_layer_1_multi_head_att_query_fc_w_0_array)),
                                "bert.encoder.layer.1.attention.self.query.bias": torch.from_numpy(encoder_layer_1_multi_head_att_query_fc_b_0_array),
                                "bert.encoder.layer.1.attention.self.key.weight": torch.from_numpy(np.transpose(encoder_layer_1_multi_head_att_key_fc_w_0_array)),
                                "bert.encoder.layer.1.attention.self.key.bias": torch.from_numpy(encoder_layer_1_multi_head_att_key_fc_b_0_array),
                                "bert.encoder.layer.1.attention.self.value.weight": torch.from_numpy(np.transpose(encoder_layer_1_multi_head_att_value_fc_w_0_array)),
                                "bert.encoder.layer.1.attention.self.value.bias": torch.from_numpy(encoder_layer_1_multi_head_att_value_fc_b_0_array),
                                "bert.encoder.layer.1.attention.output.dense.weight": torch.from_numpy(np.transpose(encoder_layer_1_multi_head_att_output_fc_w_0_array)),
                                "bert.encoder.layer.1.attention.output.dense.bias": torch.from_numpy(encoder_layer_1_multi_head_att_output_fc_b_0_array),
                                "bert.encoder.layer.1.attention.output.LayerNorm.weight": torch.from_numpy(encoder_layer_1_post_att_layer_norm_scale_array),
                                "bert.encoder.layer.1.attention.output.LayerNorm.bias": torch.from_numpy(encoder_layer_1_post_att_layer_norm_bias_array),
                                "bert.encoder.layer.1.intermediate.dense.weight": torch.from_numpy(np.transpose(encoder_layer_1_ffn_fc_0_w_0_array)),
                                "bert.encoder.layer.1.intermediate.dense.bias": torch.from_numpy(encoder_layer_1_ffn_fc_0_b_0_array),
                                "bert.encoder.layer.1.output.dense.weight": torch.from_numpy(np.transpose(encoder_layer_1_ffn_fc_1_w_0_array)),
                                "bert.encoder.layer.1.output.dense.bias": torch.from_numpy(encoder_layer_1_ffn_fc_1_b_0_array),
                                "bert.encoder.layer.1.output.LayerNorm.weight": torch.from_numpy(encoder_layer_1_post_ffn_layer_norm_scale_array),
                                "bert.encoder.layer.1.output.LayerNorm.bias": torch.from_numpy(encoder_layer_1_post_ffn_layer_norm_bias_array),

                                "bert.encoder.layer.2.attention.self.query.weight": torch.from_numpy(np.transpose(encoder_layer_2_multi_head_att_query_fc_w_0_array)),
                                "bert.encoder.layer.2.attention.self.query.bias": torch.from_numpy(encoder_layer_2_multi_head_att_query_fc_b_0_array),
                                "bert.encoder.layer.2.attention.self.key.weight": torch.from_numpy(np.transpose(encoder_layer_2_multi_head_att_key_fc_w_0_array)),
                                "bert.encoder.layer.2.attention.self.key.bias": torch.from_numpy(encoder_layer_2_multi_head_att_key_fc_b_0_array),
                                "bert.encoder.layer.2.attention.self.value.weight": torch.from_numpy(np.transpose(encoder_layer_2_multi_head_att_value_fc_w_0_array)),
                                "bert.encoder.layer.2.attention.self.value.bias": torch.from_numpy(encoder_layer_2_multi_head_att_value_fc_b_0_array),
                                "bert.encoder.layer.2.attention.output.dense.weight": torch.from_numpy(np.transpose(encoder_layer_2_multi_head_att_output_fc_w_0_array)),
                                "bert.encoder.layer.2.attention.output.dense.bias": torch.from_numpy(encoder_layer_2_multi_head_att_output_fc_b_0_array),
                                "bert.encoder.layer.2.attention.output.LayerNorm.weight": torch.from_numpy(encoder_layer_2_post_att_layer_norm_scale_array),
                                "bert.encoder.layer.2.attention.output.LayerNorm.bias": torch.from_numpy(encoder_layer_2_post_att_layer_norm_bias_array),
                                "bert.encoder.layer.2.intermediate.dense.weight": torch.from_numpy(np.transpose(encoder_layer_2_ffn_fc_0_w_0_array)),
                                "bert.encoder.layer.2.intermediate.dense.bias": torch.from_numpy(encoder_layer_2_ffn_fc_0_b_0_array),
                                "bert.encoder.layer.2.output.dense.weight": torch.from_numpy(np.transpose(encoder_layer_2_ffn_fc_1_w_0_array)),
                                "bert.encoder.layer.2.output.dense.bias": torch.from_numpy(encoder_layer_2_ffn_fc_1_b_0_array),
                                "bert.encoder.layer.2.output.LayerNorm.weight": torch.from_numpy(encoder_layer_2_post_ffn_layer_norm_scale_array),
                                "bert.encoder.layer.2.output.LayerNorm.bias": torch.from_numpy(encoder_layer_2_post_ffn_layer_norm_bias_array),

                                "bert.encoder.layer.3.attention.self.query.weight": torch.from_numpy(np.transpose(encoder_layer_3_multi_head_att_query_fc_w_0_array)),
                                "bert.encoder.layer.3.attention.self.query.bias": torch.from_numpy(encoder_layer_3_multi_head_att_query_fc_b_0_array),
                                "bert.encoder.layer.3.attention.self.key.weight": torch.from_numpy(np.transpose(encoder_layer_3_multi_head_att_key_fc_w_0_array)),
                                "bert.encoder.layer.3.attention.self.key.bias": torch.from_numpy(encoder_layer_3_multi_head_att_key_fc_b_0_array),
                                "bert.encoder.layer.3.attention.self.value.weight": torch.from_numpy(np.transpose(encoder_layer_3_multi_head_att_value_fc_w_0_array)),
                                "bert.encoder.layer.3.attention.self.value.bias": torch.from_numpy(encoder_layer_3_multi_head_att_value_fc_b_0_array),
                                "bert.encoder.layer.3.attention.output.dense.weight": torch.from_numpy(np.transpose(encoder_layer_3_multi_head_att_output_fc_w_0_array)),
                                "bert.encoder.layer.3.attention.output.dense.bias": torch.from_numpy(encoder_layer_3_multi_head_att_output_fc_b_0_array),
                                "bert.encoder.layer.3.attention.output.LayerNorm.weight": torch.from_numpy(encoder_layer_3_post_att_layer_norm_scale_array),
                                "bert.encoder.layer.3.attention.output.LayerNorm.bias": torch.from_numpy(encoder_layer_3_post_att_layer_norm_bias_array),
                                "bert.encoder.layer.3.intermediate.dense.weight": torch.from_numpy(np.transpose(encoder_layer_3_ffn_fc_0_w_0_array)),
                                "bert.encoder.layer.3.intermediate.dense.bias": torch.from_numpy(encoder_layer_3_ffn_fc_0_b_0_array),
                                "bert.encoder.layer.3.output.dense.weight": torch.from_numpy(np.transpose(encoder_layer_3_ffn_fc_1_w_0_array)),
                                "bert.encoder.layer.3.output.dense.bias": torch.from_numpy(encoder_layer_3_ffn_fc_1_b_0_array),
                                "bert.encoder.layer.3.output.LayerNorm.weight": torch.from_numpy(encoder_layer_3_post_ffn_layer_norm_scale_array),
                                "bert.encoder.layer.3.output.LayerNorm.bias": torch.from_numpy(encoder_layer_3_post_ffn_layer_norm_bias_array),

                                "bert.encoder.layer.4.attention.self.query.weight": torch.from_numpy(np.transpose(encoder_layer_4_multi_head_att_query_fc_w_0_array)),
                                "bert.encoder.layer.4.attention.self.query.bias": torch.from_numpy(encoder_layer_4_multi_head_att_query_fc_b_0_array),
                                "bert.encoder.layer.4.attention.self.key.weight": torch.from_numpy(np.transpose(encoder_layer_4_multi_head_att_key_fc_w_0_array)),
                                "bert.encoder.layer.4.attention.self.key.bias": torch.from_numpy(encoder_layer_4_multi_head_att_key_fc_b_0_array),
                                "bert.encoder.layer.4.attention.self.value.weight": torch.from_numpy(np.transpose(encoder_layer_4_multi_head_att_value_fc_w_0_array)),
                                "bert.encoder.layer.4.attention.self.value.bias": torch.from_numpy(encoder_layer_4_multi_head_att_value_fc_b_0_array),
                                "bert.encoder.layer.4.attention.output.dense.weight": torch.from_numpy(np.transpose(encoder_layer_4_multi_head_att_output_fc_w_0_array)),
                                "bert.encoder.layer.4.attention.output.dense.bias": torch.from_numpy(encoder_layer_4_multi_head_att_output_fc_b_0_array),
                                "bert.encoder.layer.4.attention.output.LayerNorm.weight": torch.from_numpy(encoder_layer_4_post_att_layer_norm_scale_array),
                                "bert.encoder.layer.4.attention.output.LayerNorm.bias": torch.from_numpy(encoder_layer_4_post_att_layer_norm_bias_array),
                                "bert.encoder.layer.4.intermediate.dense.weight": torch.from_numpy(np.transpose(encoder_layer_4_ffn_fc_0_w_0_array)),
                                "bert.encoder.layer.4.intermediate.dense.bias": torch.from_numpy(encoder_layer_4_ffn_fc_0_b_0_array),
                                "bert.encoder.layer.4.output.dense.weight": torch.from_numpy(np.transpose(encoder_layer_4_ffn_fc_1_w_0_array)),
                                "bert.encoder.layer.4.output.dense.bias": torch.from_numpy(encoder_layer_4_ffn_fc_1_b_0_array),
                                "bert.encoder.layer.4.output.LayerNorm.weight": torch.from_numpy(encoder_layer_4_post_ffn_layer_norm_scale_array),
                                "bert.encoder.layer.4.output.LayerNorm.bias": torch.from_numpy(encoder_layer_4_post_ffn_layer_norm_bias_array),

                                "bert.encoder.layer.5.attention.self.query.weight": torch.from_numpy(np.transpose(encoder_layer_5_multi_head_att_query_fc_w_0_array)),
                                "bert.encoder.layer.5.attention.self.query.bias": torch.from_numpy(encoder_layer_5_multi_head_att_query_fc_b_0_array),
                                "bert.encoder.layer.5.attention.self.key.weight": torch.from_numpy(np.transpose(encoder_layer_5_multi_head_att_key_fc_w_0_array)),
                                "bert.encoder.layer.5.attention.self.key.bias": torch.from_numpy(encoder_layer_5_multi_head_att_key_fc_b_0_array),
                                "bert.encoder.layer.5.attention.self.value.weight": torch.from_numpy(np.transpose(encoder_layer_5_multi_head_att_value_fc_w_0_array)),
                                "bert.encoder.layer.5.attention.self.value.bias": torch.from_numpy(encoder_layer_5_multi_head_att_value_fc_b_0_array),
                                "bert.encoder.layer.5.attention.output.dense.weight": torch.from_numpy(np.transpose(encoder_layer_5_multi_head_att_output_fc_w_0_array)),
                                "bert.encoder.layer.5.attention.output.dense.bias": torch.from_numpy(encoder_layer_5_multi_head_att_output_fc_b_0_array),
                                "bert.encoder.layer.5.attention.output.LayerNorm.weight": torch.from_numpy(encoder_layer_5_post_att_layer_norm_scale_array),
                                "bert.encoder.layer.5.attention.output.LayerNorm.bias": torch.from_numpy(encoder_layer_5_post_att_layer_norm_bias_array),
                                "bert.encoder.layer.5.intermediate.dense.weight": torch.from_numpy(np.transpose(encoder_layer_5_ffn_fc_0_w_0_array)),
                                "bert.encoder.layer.5.intermediate.dense.bias": torch.from_numpy(encoder_layer_5_ffn_fc_0_b_0_array),
                                "bert.encoder.layer.5.output.dense.weight": torch.from_numpy(np.transpose(encoder_layer_5_ffn_fc_1_w_0_array)),
                                "bert.encoder.layer.5.output.dense.bias": torch.from_numpy(encoder_layer_5_ffn_fc_1_b_0_array),
                                "bert.encoder.layer.5.output.LayerNorm.weight": torch.from_numpy(encoder_layer_5_post_ffn_layer_norm_scale_array),
                                "bert.encoder.layer.5.output.LayerNorm.bias": torch.from_numpy(encoder_layer_5_post_ffn_layer_norm_bias_array),

                                "bert.encoder.layer.6.attention.self.query.weight": torch.from_numpy(np.transpose(encoder_layer_6_multi_head_att_query_fc_w_0_array)),
                                "bert.encoder.layer.6.attention.self.query.bias": torch.from_numpy(encoder_layer_6_multi_head_att_query_fc_b_0_array),
                                "bert.encoder.layer.6.attention.self.key.weight": torch.from_numpy(np.transpose(encoder_layer_6_multi_head_att_key_fc_w_0_array)),
                                "bert.encoder.layer.6.attention.self.key.bias": torch.from_numpy(encoder_layer_6_multi_head_att_key_fc_b_0_array),
                                "bert.encoder.layer.6.attention.self.value.weight": torch.from_numpy(np.transpose(encoder_layer_6_multi_head_att_value_fc_w_0_array)),
                                "bert.encoder.layer.6.attention.self.value.bias": torch.from_numpy(encoder_layer_6_multi_head_att_value_fc_b_0_array),
                                "bert.encoder.layer.6.attention.output.dense.weight": torch.from_numpy(np.transpose(encoder_layer_6_multi_head_att_output_fc_w_0_array)),
                                "bert.encoder.layer.6.attention.output.dense.bias": torch.from_numpy(encoder_layer_6_multi_head_att_output_fc_b_0_array),
                                "bert.encoder.layer.6.attention.output.LayerNorm.weight": torch.from_numpy(encoder_layer_6_post_att_layer_norm_scale_array),
                                "bert.encoder.layer.6.attention.output.LayerNorm.bias": torch.from_numpy(encoder_layer_6_post_att_layer_norm_bias_array),
                                "bert.encoder.layer.6.intermediate.dense.weight": torch.from_numpy(np.transpose(encoder_layer_6_ffn_fc_0_w_0_array)),
                                "bert.encoder.layer.6.intermediate.dense.bias": torch.from_numpy(encoder_layer_6_ffn_fc_0_b_0_array),
                                "bert.encoder.layer.6.output.dense.weight": torch.from_numpy(np.transpose(encoder_layer_6_ffn_fc_1_w_0_array)),
                                "bert.encoder.layer.6.output.dense.bias": torch.from_numpy(encoder_layer_6_ffn_fc_1_b_0_array),
                                "bert.encoder.layer.6.output.LayerNorm.weight": torch.from_numpy(encoder_layer_6_post_ffn_layer_norm_scale_array),
                                "bert.encoder.layer.6.output.LayerNorm.bias": torch.from_numpy(encoder_layer_6_post_ffn_layer_norm_bias_array),

                                "bert.encoder.layer.7.attention.self.query.weight": torch.from_numpy(np.transpose(encoder_layer_7_multi_head_att_query_fc_w_0_array)),
                                "bert.encoder.layer.7.attention.self.query.bias": torch.from_numpy(encoder_layer_7_multi_head_att_query_fc_b_0_array),
                                "bert.encoder.layer.7.attention.self.key.weight": torch.from_numpy(np.transpose(encoder_layer_7_multi_head_att_key_fc_w_0_array)),
                                "bert.encoder.layer.7.attention.self.key.bias": torch.from_numpy(encoder_layer_7_multi_head_att_key_fc_b_0_array),
                                "bert.encoder.layer.7.attention.self.value.weight": torch.from_numpy(np.transpose(encoder_layer_7_multi_head_att_value_fc_w_0_array)),
                                "bert.encoder.layer.7.attention.self.value.bias": torch.from_numpy(encoder_layer_7_multi_head_att_value_fc_b_0_array),
                                "bert.encoder.layer.7.attention.output.dense.weight": torch.from_numpy(np.transpose(encoder_layer_7_multi_head_att_output_fc_w_0_array)),
                                "bert.encoder.layer.7.attention.output.dense.bias": torch.from_numpy(encoder_layer_7_multi_head_att_output_fc_b_0_array),
                                "bert.encoder.layer.7.attention.output.LayerNorm.weight": torch.from_numpy(encoder_layer_7_post_att_layer_norm_scale_array),
                                "bert.encoder.layer.7.attention.output.LayerNorm.bias": torch.from_numpy(encoder_layer_7_post_att_layer_norm_bias_array),
                                "bert.encoder.layer.7.intermediate.dense.weight": torch.from_numpy(np.transpose(encoder_layer_7_ffn_fc_0_w_0_array)),
                                "bert.encoder.layer.7.intermediate.dense.bias": torch.from_numpy(encoder_layer_7_ffn_fc_0_b_0_array),
                                "bert.encoder.layer.7.output.dense.weight": torch.from_numpy(np.transpose(encoder_layer_7_ffn_fc_1_w_0_array)),
                                "bert.encoder.layer.7.output.dense.bias": torch.from_numpy(encoder_layer_7_ffn_fc_1_b_0_array),
                                "bert.encoder.layer.7.output.LayerNorm.weight": torch.from_numpy(encoder_layer_7_post_ffn_layer_norm_scale_array),
                                "bert.encoder.layer.7.output.LayerNorm.bias": torch.from_numpy(encoder_layer_7_post_ffn_layer_norm_bias_array),

                                "bert.encoder.layer.8.attention.self.query.weight": torch.from_numpy(np.transpose(encoder_layer_8_multi_head_att_query_fc_w_0_array)),
                                "bert.encoder.layer.8.attention.self.query.bias": torch.from_numpy(encoder_layer_8_multi_head_att_query_fc_b_0_array),
                                "bert.encoder.layer.8.attention.self.key.weight": torch.from_numpy(np.transpose(encoder_layer_8_multi_head_att_key_fc_w_0_array)),
                                "bert.encoder.layer.8.attention.self.key.bias": torch.from_numpy(encoder_layer_8_multi_head_att_key_fc_b_0_array),
                                "bert.encoder.layer.8.attention.self.value.weight": torch.from_numpy(np.transpose(encoder_layer_8_multi_head_att_value_fc_w_0_array)),
                                "bert.encoder.layer.8.attention.self.value.bias": torch.from_numpy(encoder_layer_8_multi_head_att_value_fc_b_0_array),
                                "bert.encoder.layer.8.attention.output.dense.weight": torch.from_numpy(np.transpose(encoder_layer_8_multi_head_att_output_fc_w_0_array)),
                                "bert.encoder.layer.8.attention.output.dense.bias": torch.from_numpy(encoder_layer_8_multi_head_att_output_fc_b_0_array),
                                "bert.encoder.layer.8.attention.output.LayerNorm.weight": torch.from_numpy(encoder_layer_8_post_att_layer_norm_scale_array),
                                "bert.encoder.layer.8.attention.output.LayerNorm.bias": torch.from_numpy(encoder_layer_8_post_att_layer_norm_bias_array),
                                "bert.encoder.layer.8.intermediate.dense.weight": torch.from_numpy(np.transpose(encoder_layer_8_ffn_fc_0_w_0_array)),
                                "bert.encoder.layer.8.intermediate.dense.bias": torch.from_numpy(encoder_layer_8_ffn_fc_0_b_0_array),
                                "bert.encoder.layer.8.output.dense.weight": torch.from_numpy(np.transpose(encoder_layer_8_ffn_fc_1_w_0_array)),
                                "bert.encoder.layer.8.output.dense.bias": torch.from_numpy(encoder_layer_8_ffn_fc_1_b_0_array),
                                "bert.encoder.layer.8.output.LayerNorm.weight": torch.from_numpy(encoder_layer_8_post_ffn_layer_norm_scale_array),
                                "bert.encoder.layer.8.output.LayerNorm.bias": torch.from_numpy(encoder_layer_8_post_ffn_layer_norm_bias_array),

                                "bert.encoder.layer.9.attention.self.query.weight": torch.from_numpy(np.transpose(encoder_layer_9_multi_head_att_query_fc_w_0_array)),
                                "bert.encoder.layer.9.attention.self.query.bias": torch.from_numpy(encoder_layer_9_multi_head_att_query_fc_b_0_array),
                                "bert.encoder.layer.9.attention.self.key.weight": torch.from_numpy(np.transpose(encoder_layer_9_multi_head_att_key_fc_w_0_array)),
                                "bert.encoder.layer.9.attention.self.key.bias": torch.from_numpy(encoder_layer_9_multi_head_att_key_fc_b_0_array),
                                "bert.encoder.layer.9.attention.self.value.weight": torch.from_numpy(np.transpose(encoder_layer_9_multi_head_att_value_fc_w_0_array)),
                                "bert.encoder.layer.9.attention.self.value.bias": torch.from_numpy(encoder_layer_9_multi_head_att_value_fc_b_0_array),
                                "bert.encoder.layer.9.attention.output.dense.weight": torch.from_numpy(np.transpose(encoder_layer_9_multi_head_att_output_fc_w_0_array)),
                                "bert.encoder.layer.9.attention.output.dense.bias": torch.from_numpy(encoder_layer_9_multi_head_att_output_fc_b_0_array),
                                "bert.encoder.layer.9.attention.output.LayerNorm.weight": torch.from_numpy(encoder_layer_9_post_att_layer_norm_scale_array),
                                "bert.encoder.layer.9.attention.output.LayerNorm.bias": torch.from_numpy(encoder_layer_9_post_att_layer_norm_bias_array),
                                "bert.encoder.layer.9.intermediate.dense.weight": torch.from_numpy(np.transpose(encoder_layer_9_ffn_fc_0_w_0_array)),
                                "bert.encoder.layer.9.intermediate.dense.bias": torch.from_numpy(encoder_layer_9_ffn_fc_0_b_0_array),
                                "bert.encoder.layer.9.output.dense.weight": torch.from_numpy(np.transpose(encoder_layer_9_ffn_fc_1_w_0_array)),
                                "bert.encoder.layer.9.output.dense.bias": torch.from_numpy(encoder_layer_9_ffn_fc_1_b_0_array),
                                "bert.encoder.layer.9.output.LayerNorm.weight": torch.from_numpy(encoder_layer_9_post_ffn_layer_norm_scale_array),
                                "bert.encoder.layer.9.output.LayerNorm.bias": torch.from_numpy(encoder_layer_9_post_ffn_layer_norm_bias_array),

                                "bert.encoder.layer.10.attention.self.query.weight": torch.from_numpy(np.transpose(encoder_layer_10_multi_head_att_query_fc_w_0_array)),
                                "bert.encoder.layer.10.attention.self.query.bias": torch.from_numpy(encoder_layer_10_multi_head_att_query_fc_b_0_array),
                                "bert.encoder.layer.10.attention.self.key.weight": torch.from_numpy(np.transpose(encoder_layer_10_multi_head_att_key_fc_w_0_array)),
                                "bert.encoder.layer.10.attention.self.key.bias": torch.from_numpy(encoder_layer_10_multi_head_att_key_fc_b_0_array),
                                "bert.encoder.layer.10.attention.self.value.weight": torch.from_numpy(np.transpose(encoder_layer_10_multi_head_att_value_fc_w_0_array)),
                                "bert.encoder.layer.10.attention.self.value.bias": torch.from_numpy(encoder_layer_10_multi_head_att_value_fc_b_0_array),
                                "bert.encoder.layer.10.attention.output.dense.weight": torch.from_numpy(np.transpose(encoder_layer_10_multi_head_att_output_fc_w_0_array)),
                                "bert.encoder.layer.10.attention.output.dense.bias": torch.from_numpy(encoder_layer_10_multi_head_att_output_fc_b_0_array),
                                "bert.encoder.layer.10.attention.output.LayerNorm.weight": torch.from_numpy(encoder_layer_10_post_att_layer_norm_scale_array),
                                "bert.encoder.layer.10.attention.output.LayerNorm.bias": torch.from_numpy(encoder_layer_10_post_att_layer_norm_bias_array),
                                "bert.encoder.layer.10.intermediate.dense.weight": torch.from_numpy(np.transpose(encoder_layer_10_ffn_fc_0_w_0_array)),
                                "bert.encoder.layer.10.intermediate.dense.bias": torch.from_numpy(encoder_layer_10_ffn_fc_0_b_0_array),
                                "bert.encoder.layer.10.output.dense.weight": torch.from_numpy(np.transpose(encoder_layer_10_ffn_fc_1_w_0_array)),
                                "bert.encoder.layer.10.output.dense.bias": torch.from_numpy(encoder_layer_10_ffn_fc_1_b_0_array),
                                "bert.encoder.layer.10.output.LayerNorm.weight": torch.from_numpy(encoder_layer_10_post_ffn_layer_norm_scale_array),
                                "bert.encoder.layer.10.output.LayerNorm.bias": torch.from_numpy(encoder_layer_10_post_ffn_layer_norm_bias_array),

                                "bert.encoder.layer.11.attention.self.query.weight": torch.from_numpy(np.transpose(encoder_layer_11_multi_head_att_query_fc_w_0_array)),
                                "bert.encoder.layer.11.attention.self.query.bias": torch.from_numpy(encoder_layer_11_multi_head_att_query_fc_b_0_array),
                                "bert.encoder.layer.11.attention.self.key.weight": torch.from_numpy(np.transpose(encoder_layer_11_multi_head_att_key_fc_w_0_array)),
                                "bert.encoder.layer.11.attention.self.key.bias": torch.from_numpy(encoder_layer_11_multi_head_att_key_fc_b_0_array),
                                "bert.encoder.layer.11.attention.self.value.weight": torch.from_numpy(np.transpose(encoder_layer_11_multi_head_att_value_fc_w_0_array)),
                                "bert.encoder.layer.11.attention.self.value.bias": torch.from_numpy(encoder_layer_11_multi_head_att_value_fc_b_0_array),
                                "bert.encoder.layer.11.attention.output.dense.weight": torch.from_numpy(np.transpose(encoder_layer_11_multi_head_att_output_fc_w_0_array)),
                                "bert.encoder.layer.11.attention.output.dense.bias": torch.from_numpy(encoder_layer_11_multi_head_att_output_fc_b_0_array),
                                "bert.encoder.layer.11.attention.output.LayerNorm.weight": torch.from_numpy(encoder_layer_11_post_att_layer_norm_scale_array),
                                "bert.encoder.layer.11.attention.output.LayerNorm.bias": torch.from_numpy(encoder_layer_11_post_att_layer_norm_bias_array),
                                "bert.encoder.layer.11.intermediate.dense.weight": torch.from_numpy(np.transpose(encoder_layer_11_ffn_fc_0_w_0_array)),
                                "bert.encoder.layer.11.intermediate.dense.bias": torch.from_numpy(encoder_layer_11_ffn_fc_0_b_0_array),
                                "bert.encoder.layer.11.output.dense.weight": torch.from_numpy(np.transpose(encoder_layer_11_ffn_fc_1_w_0_array)),
                                "bert.encoder.layer.11.output.dense.bias": torch.from_numpy(encoder_layer_11_ffn_fc_1_b_0_array),
                                "bert.encoder.layer.11.output.LayerNorm.weight": torch.from_numpy(encoder_layer_11_post_ffn_layer_norm_scale_array),
                                "bert.encoder.layer.11.output.LayerNorm.bias": torch.from_numpy(encoder_layer_11_post_ffn_layer_norm_bias_array),

                                "bert.pooler.dense.weight": torch.from_numpy(np.transpose(pooled_fc_w_0_array)),
                                "bert.pooler.dense.bias": torch.from_numpy(pooled_fc_b_0_array),

                                "cls.predictions.bias": torch.from_numpy(mask_lm_out_fc_b_0_array),
                                "cls.predictions.transform.dense.weight": torch.from_numpy(np.transpose(mask_lm_trans_fc_w_0_array)),
                                "cls.predictions.transform.dense.bias": torch.from_numpy(mask_lm_trans_fc_b_0_array),
                                "cls.predictions.transform.LayerNorm.weight": torch.from_numpy(mask_lm_trans_layer_norm_scale_array),
                                "cls.predictions.transform.LayerNorm.bias": torch.from_numpy(mask_lm_trans_layer_norm_bias_array),
                                "cls.predictions.decoder.weight": model.state_dict()["cls.predictions.decoder.weight"]

                                })
model.load_state_dict(new_state_dict)
#print(model.state_dict()["cls.predictions.transform.LayerNorm.bias"])

# Save pytorch model
print("Save PyTorch Model to {}".format("./pytorch/pytorch_model.bin"))
torch.save(model.state_dict(), "./pytorch/pytorch_model.bin")
