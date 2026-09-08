import inspect
from ultralytics.nn.modules.head import Detect
print(inspect.getsource(Detect.forward))
if hasattr(Detect,'forward_head'):print(inspect.getsource(Detect.forward_head))
