from transformers import AutoConfig

def get_config(pretrained_path=None, cmd_args=None):
    config = AutoConfig.from_pretrained(pretrained_path)

    # update pretrained config with our Argparse config.
    for k in cmd_args.__dict__:
        config.__dict__[k] = cmd_args.__dict__[k]

    # return
    return config