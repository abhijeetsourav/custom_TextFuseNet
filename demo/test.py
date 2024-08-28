import importlib.util
import sys
import os

module_name = "visualizer"
file_path = os.path.join(os.path.dirname((os.path.dirname(__file__))), 'detectron2', 'utils', 'visualizer.py')
print(file_path)

spec   = importlib.util.spec_from_file_location(module_name, file_path)
module = importlib.util.module_from_spec(spec)
sys.modules[module_name] = module
spec.loader.exec_module(module)   



# Now you can use the module
# module.some_function()



# Define a custom loader
# class CustomLoader(importlib.abc.Loader):
#     def load_module(self, fullname):

#         # Load the module using the default import mechanism
#         module = importlib.util.find_spec(fullname).loader.load_module()

#         # Add custom logic to the import process
#         print(f"Module {fullname} has been imported")
#         return module

# # Use the custom loader
# loader = CustomLoader()
# module = loader.load_module("math")
# result = module.sqrt(16)
# print(result)