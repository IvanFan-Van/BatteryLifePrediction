"""Environment passed to queued training processes."""
import os


def clean_environment():
    """Do not copy API credentials into Pueue's persisted task environment."""
    allowed = {'PATH','PATHEXT','SYSTEMROOT','WINDIR','COMSPEC','USERPROFILE','APPDATA',
               'LOCALAPPDATA','TEMP','TMP','USERNAME','USERDOMAIN','PROCESSOR_ARCHITECTURE',
               'NUMBER_OF_PROCESSORS','PROGRAMDATA','PROGRAMFILES','PROGRAMFILES(X86)',
               'COMMONPROGRAMFILES','VIRTUAL_ENV','UV_CACHE_DIR','UV_INTERNAL__PYTHONHOME',
               'PYTHONUTF8','PYTHONIOENCODING','PUEUE_CONFIG_PATH'}
    environment = {k:v for k,v in os.environ.items() if k.upper() in allowed}
    environment.update(WANDB_MODE='offline', WANDB_SILENT='true',
                       CUDA_VISIBLE_DEVICES=os.environ.get('CUDA_VISIBLE_DEVICES', '0' if os.name=='nt' else '2'),
                       CUBLAS_WORKSPACE_CONFIG=':4096:8',
                       PYTHONUTF8='1', PYTHONIOENCODING='utf-8')
    return environment
