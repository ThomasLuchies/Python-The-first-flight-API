import yaml
from IPython.core.magic import register_line_cell_magic

with open("data.yaml", 'r') as stream:
    num_classes = str(yaml.safe_load(stream)['nc'])

@register_line_cell_magic
def writetemplate(line, cell):
    with open(line, 'w') as f:
        f.write(cell.format(**globals()))