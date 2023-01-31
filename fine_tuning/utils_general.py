import os

def reformat_model_path(x, args):
    fp_marker = './'
    if (os.environ.get('env') == 'bb' or getattr(args, 'env', 'local') == 'bb') and (not x.startswith(fp_marker)):
        return os.path.join(fp_marker, x)
    else:
        return x


def check_and_install_ninja():
    import sys
    sys.path.append('/job/.local/bin')

    # check for ninja
    from torch.utils.cpp_extension import is_ninja_available
    if not is_ninja_available():
        print('No ninja found, attempting to install now...')
        import subprocess
        subprocess.run('python -m pip install ninja', shell=True)
