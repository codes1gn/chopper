# Auto Generate Documentation for Chopper

To make easy to read and use the Chopper project, there use Sphinx library to automatically generate description and function interface documentation for the project.

## Quick Generate

### Requirement
```
cd docs/
pip install -r requirements.txt
```

### Generate documents

#### Chopper frontend API documents
```shell
cd docs/
sphinx-apidoc -f -o ./source/ ../chopper_frontend
```
After, will auto generate frontend api as flow:
```
Creating file ./source/chopper_frontend.rst.
Creating file ./source/chopper_frontend.numpy.rst.
Creating file ./source/chopper_frontend.python.rst.
Creating file ./source/chopper_frontend.scaffold.rst.
Creating file ./source/chopper_frontend.scaffold.mlir_dialects.rst.
Creating file ./source/chopper_frontend.scaffold.utils.rst.
Creating file ./source/chopper_frontend.torch.rst.
Creating file ./source/modules.rst.
```
The modules.rst is main entrance

#### Chopper documents 
This part of the documentation is actually generated automatically during the project build phase, more detail refer `./build/docs`, there only copy the .md file and reorganize. Additionly, we use the `./docs/source/md2rst.py` to convert .md file to .rst file.

Now, we only to run:
```
sphinx-autobuild -b html source/ build/
```
