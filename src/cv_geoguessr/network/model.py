import torch
from torch import nn
from torchsummary import summary
from collections import OrderedDict


def model_layers(model, input_size, batch_size=-1, device="cuda"):
    """Returns the layers of the model flattened, based on the summary code"""

    def register_hook(module):

        def hook(module, input, output):
            class_name = str(module.__class__).split(".")[-1].split("'")[0]
            module_idx = len(layers)

            m_key = "%s-%i" % (class_name, module_idx + 1)
            layers[m_key] = OrderedDict()
            layers[m_key]["input_shape"] = list(input[0].size())
            layers[m_key]["input_shape"][0] = batch_size
            if isinstance(output, (list, tuple)):
                layers[m_key]["output_shape"] = [
                    [-1] + list(o.size())[1:] for o in output
                ]
            else:
                layers[m_key]["output_shape"] = list(output.size())
                layers[m_key]["output_shape"][0] = batch_size

            params = 0
            layers[m_key]["params"] = module.parameters()
            if hasattr(module, "weight") and hasattr(module.weight, "size"):
                params += torch.prod(torch.LongTensor(list(module.weight.size())))
                layers[m_key]["trainable"] = module.weight.requires_grad
                layers[m_key]["weights"] = module.weight
            if hasattr(module, "bias") and hasattr(module.bias, "size"):
                params += torch.prod(torch.LongTensor(list(module.bias.size())))
            layers[m_key]["nb_params"] = params

        if (
            not isinstance(module, nn.Sequential)
            and not isinstance(module, nn.ModuleList)
            and not (module == model)
        ):
            hooks.append(module.register_forward_hook(hook))

    device = device.lower()

    assert device in [
        "cuda",
        "cpu",
    ], "Input device is not valid, please specify 'cuda' or 'cpu'"

    if device == "cuda" and torch.cuda.is_available():
        dtype = torch.cuda.FloatTensor
    else:
        dtype = torch.FloatTensor

    # multiple inputs to the network
    if isinstance(input_size, tuple):
        input_size = [input_size]

    # batch_size of 2 for batchnorm
    x = [torch.rand(2, *in_size).type(dtype) for in_size in input_size]
    # print(type(x[0]))

    # create properties
    layers = OrderedDict()
    hooks = []

    # register hook
    model.apply(register_hook)

    # make a forward pass
    # print(x.shape)
    model(*x)

    # remove these hooks
    for h in hooks:
        h.remove()

    return layers


def lock_layers(model, input_shape, lock_factor):
    layers = model_layers(model, input_shape)
    n = len(layers)
    locked = 0
    ll = 0
    x = 0
    # print(n)  # 10 layers
    # print(model_layers)

    for i, layer in enumerate(layers):
        # for layer_param in layer.parameters():
        #     layer_param.requires_grad = i > n * lock_factor
        for param in layers[layer]["params"]:
            x += 1

            param.requires_grad = i > n * lock_factor

            if i <= n * lock_factor:
                locked += 1
        if i <= n * lock_factor:
            ll += 1

    print(f"{ll}/{n} layers locked")
    print(f"{locked}/{x} params locked")

    return model


# load an old model
def load_model(model, PATH, lock_factor, device):
    checkpoint = torch.load(PATH)
    model.load_state_dict(checkpoint)
    model = lock_layers(model, (3, 224, 224), lock_factor)
    model.to(device)

    return model