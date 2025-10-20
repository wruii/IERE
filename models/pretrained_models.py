import os
import math
import torch
# from timm.models.helpers import adapt_input_conv


pretrained_model_zoo = {
    "prostate_a_to_b.ckpt":{
        "top_level_dict_key": 'state_dict',
        "prefix_to_remove": 'model.'
    },
    "prostate_b_to_a.ckpt":{
        "top_level_dict_key": 'state_dict',
        "prefix_to_remove": 'model.'
    },
}

def prefix_get(dict, querry, default=None):
    """ Dictionary lookup, but exact key matching is not required, if the querry starts with a key, it's a match.
    """
    value = default
    for k in dict.keys():
        if querry.startswith(k):
            value = dict[k]
    return value

def get_state_dict_form_pretrained_model_zoo(model_name:str, in_chans:int, default_models_folder='models', prefix_to_add=""):
    pretrained_config = prefix_get(pretrained_model_zoo, os.path.relpath(model_name))
    if pretrained_config is None:
        print(f"Model {model_name} not found in pretrained_model_zoo. Available models: {pretrained_model_zoo.keys()}")
        return None
    if os.path.exists(model_name):
        model_path = model_name
    elif pretrained_config.get("path", None) is None:
        model_path = os.path.join(default_models_folder, model_name)
    else:
        model_path = os.path.join(pretrained_config["path"], model_name)
    try:
        pretrained_weights = torch.load(model_path, map_location=torch.device('cpu'))
    except:
        pretrained_weights = torch.load(os.path.join('../models', model_name), map_location=torch.device('cpu'))
    pretrained_weights = modify_state_dict(pretrained_weights,
                                           pretrained_config.get('top_level_dict_key', None),
                                           pretrained_config.get('submodules_to_drop', None),
                                           pretrained_config.get("prefix_to_remove", ''),
                                           prefix_to_add)

    pretrained_weights = prepare_pretrained_input_weights(pretrained_weights, in_chans, prefix_to_add)

    print(f"State dict loaded from {model_path}")
    return pretrained_weights

def get_state_dict_form_pretrained_model_label(model_name:str,  default_models_folder='models'):
    pretrained_config = prefix_get(pretrained_model_zoo, os.path.relpath(model_name))
    if pretrained_config is None:
        print(f"Model {model_name} not found in pretrained_model_zoo. Available models: {pretrained_model_zoo.keys()}")
        return None
    if os.path.exists(model_name):
        model_path = model_name
    elif pretrained_config.get("path", None) is None:
        model_path = os.path.join(default_models_folder, model_name)
    else:
        model_path = os.path.join(pretrained_config["path"], model_name)
    try:
        pretrained_weights = torch.load(model_path, map_location=torch.device('cpu'))
    except:
        model_path = os.path.join('checkpoints', model_name) # label
        try:
            pretrained_weights = torch.load(model_path, map_location=torch.device('cpu'))
        except:
            model_path = os.path.join('prostate', model_path)  # final finetuned
            pretrained_weights = torch.load(model_path, map_location=torch.device('cpu'))
    encoder_weights = modify_state_dict_label(pretrained_weights,
                                              pretrained_config.get('top_level_dict_key', None),
                                              pretrained_config.get('submodules_to_drop', None),
                                              'model.encoder.',)
    decoder_weights = modify_state_dict_label(pretrained_weights,
                                              pretrained_config.get('top_level_dict_key', None),
                                              pretrained_config.get('submodules_to_drop', None),
                                              'model.decoder.',)
    segmentation_weights = modify_state_dict_label(pretrained_weights,
                                              pretrained_config.get('top_level_dict_key', None),
                                              pretrained_config.get('submodules_to_drop', None),
                                              'model.segmentation_head.',)
    regression_weights = modify_state_dict_label(pretrained_weights,
                                              pretrained_config.get('top_level_dict_key', None),
                                              pretrained_config.get('submodules_to_drop', None),
                                              'model.regression_head.',)

    print(f"State dict loaded from {model_path}")
    return {'encoder': encoder_weights, "decoder": decoder_weights, "regression": regression_weights,
            "segmentation": segmentation_weights}

def get_state_dict_form_pretrained_model_label_ab(model_name:str,  default_models_folder='models'):
    pretrained_config = prefix_get(pretrained_model_zoo, os.path.relpath(model_name))
    if pretrained_config is None:
        print(f"Model {model_name} not found in pretrained_model_zoo. Available models: {pretrained_model_zoo.keys()}")
        return None
    if os.path.exists(model_name):
        model_path = model_name
    elif pretrained_config.get("path", None) is None:
        model_path = os.path.join(default_models_folder, model_name)
    else:
        model_path = os.path.join(pretrained_config["path"], model_name)
    try:
        pretrained_weights = torch.load(model_path, map_location=torch.device('cpu'))
    except:
        model_path = os.path.join('checkpoints', model_name) # label
        try:
            pretrained_weights = torch.load(model_path, map_location=torch.device('cpu'))
        except:
            model_path = os.path.join('prostate', model_path)  # final finetuned
            pretrained_weights = torch.load(model_path, map_location=torch.device('cpu'))
    encoder_weights = modify_state_dict_label(pretrained_weights,
                                              pretrained_config.get('top_level_dict_key', None),
                                              pretrained_config.get('submodules_to_drop', None),
                                              'model.encoder.',)
    decoder_weights = modify_state_dict_label(pretrained_weights,
                                              pretrained_config.get('top_level_dict_key', None),
                                              pretrained_config.get('submodules_to_drop', None),
                                              'model.decoder.',)
    segmentation_weights = modify_state_dict_label(pretrained_weights,
                                              pretrained_config.get('top_level_dict_key', None),
                                              pretrained_config.get('submodules_to_drop', None),
                                              'model.segmentation_head.',)
    regression_weights = modify_state_dict_label(pretrained_weights,
                                              pretrained_config.get('top_level_dict_key', None),
                                              pretrained_config.get('submodules_to_drop', None),
                                              'model.regression_head.',)

    print(f"State dict loaded from {model_path}")
    return {'encoder': encoder_weights, "decoder": decoder_weights, "regression": regression_weights,
            "segmentation": segmentation_weights}


def modify_state_dict(state_dict, top_level_dict_key=None, submodules_to_drop=None, prefix_to_remove='model.', prefix_to_add=""):
    if top_level_dict_key not in [None, '']:
        state_dict = state_dict[top_level_dict_key]

    if submodules_to_drop not in [None, []]:
        for submodule_to_drop in submodules_to_drop:
            state_dict = {k: v for k, v in state_dict.items() if submodule_to_drop not in k}
    
    len_prefix = len(prefix_to_remove)
    in_state_dict = state_dict
    out_state_dict = {}
    for k, v in in_state_dict.items():
        if k.startswith(prefix_to_remove):
            new_key = prefix_to_add + k[len_prefix:]
            out_state_dict[new_key] = v

    return out_state_dict


def modify_state_dict_label(state_dict, top_level_dict_key=None, submodules_to_drop=None, prefix_to_remove='model.',
                      prefix_to_add=""):
    if top_level_dict_key not in [None, '']:
        state_dict = state_dict[top_level_dict_key]

    if submodules_to_drop not in [None, []]:
        for submodule_to_drop in submodules_to_drop:
            state_dict = {k: v for k, v in state_dict.items() if submodule_to_drop not in k}

    len_prefix = len(prefix_to_remove)
    in_state_dict = state_dict
    out_state_dict = {}
    for k, v in in_state_dict.items():
        if k.startswith(prefix_to_remove):
            new_key = prefix_to_add + k[len_prefix:]
            out_state_dict[new_key] = v

    return out_state_dict


def prepare_pretrained_input_weights(state_dict, in_chans=3, prefix_to_add=""):
    weight_name = prefix_to_add + 'conv1.weight' # Only supports resnets for now
    try:
        weight_in_chans = state_dict[weight_name].shape[1]
        state_dict[weight_name] = adapt_input_conv(in_chans, state_dict[weight_name])
        print(
            f'Converted input conv {weight_name} pretrained weights from {weight_in_chans} to {in_chans} channel(s)')
    except NotImplementedError as e:
        del state_dict[weight_name]
        print(
            f'Unable to convert pretrained {weight_name} weights, using random init for this layer.')
    return state_dict

def adapt_input_conv(in_chans, conv_weight):
    """
    Based on: https://github.com/rwightman/pytorch-image-models/blob/7cd4204a289cd0dddad925c076cdb96ae9670e87/timm/models/helpers.py
    """

    conv_type = conv_weight.dtype
    O, I, J, K = conv_weight.shape
    if I == in_chans:
        # No need to convert
        return conv_weight
    conv_weight = conv_weight.float()  # Some weights are in torch.half, ensure it's float for sum on CPU
    if in_chans == 1:
        # n channel input -> 1 channel input conversion
        if I > 3:
            assert conv_weight.shape[1] % 3 == 0
            # For models with space2depth stems
            conv_weight = conv_weight.reshape(O, I // 3, 3, J, K)
            conv_weight = conv_weight.sum(dim=2, keepdim=False)
        else:
            conv_weight = conv_weight.sum(dim=1, keepdim=True)
    elif in_chans != 3 and I == 3:
        # 3 channel (RGB) input -> 2, 4, 5, 6, ... n channel input conversion
        # NOTE this strategy should be better than random init, but there could be other combinations of
        # the original RGB input layer weights that'd work better for specific cases.
        repeat = int(math.ceil(in_chans / 3))
        conv_weight = conv_weight.repeat(1, repeat, 1, 1)[:, :in_chans, :, :]
        conv_weight *= (3 / float(in_chans))
    elif I == 1:
        conv_weight = conv_weight.repeat(1, in_chans, 1, 1)
        conv_weight /= float(in_chans)
    else:
        raise NotImplementedError(f'{I} -> {in_chans} channel conversion not supported.')
    conv_weight = conv_weight.to(conv_type)
    return conv_weight


