import argparse
from pathlib import Path, PurePosixPath
import glob
import json
import yaml

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('lb_folder', nargs=1, help='input label folder')
    parser.add_argument('new_lb_folder', nargs=1, help='output label folder')
    parser.add_argument('replace_scenario', nargs=1, help='class change json/yaml')
    opt = parser.parse_args()

    lb_folder = Path(opt.lb_folder[0])
    old_labels = "/" + PurePosixPath(lb_folder).parts[-1] + "/"
    new_lb_folder = "/" + opt.new_lb_folder[0] + "/"
    file_list = glob.glob(str(lb_folder / '**' / '*.txt'), recursive=True)
    replace_scenario = opt.replace_scenario[0]
    
    if Path(replace_scenario).suffix == '.json':
        with open(replace_scenario, "rt") as f:
            class_id_replace = json.load(f)
            new_class_id_replace = dict()
            for key, value in class_id_replace.items():
                try:
                    int_value = int(value)
                    new_class_id_replace[key] = value
                    #print(f'{key} - {value}')
                except:
                    pass
            class_id_replace = new_class_id_replace

    elif Path(replace_scenario).suffix == '.yaml':
        with open(replace_scenario, "rt") as f:
            replace_yaml = yaml.safe_load(f)

        skip_other_classes = replace_yaml.get('skip_other_classes', False)

        class_id_replace = {}
        for class_id, class_name in replace_yaml['names'].items():
            if class_name in replace_yaml['select']:
                class_id_replace[class_id] = replace_yaml['select'][class_name]
            else:
                if skip_other_classes:
                    class_id_replace[class_id] = -1

        class_names = replace_yaml['names']

        new_class_names_by_id={}
        for class_id, class_name in class_names.items():
            if class_id in class_id_replace:
                if class_id_replace[class_id] >=0:
                    new_id = class_id_replace[class_id]
                    new_class_names_by_id[new_id] = replace_yaml['new_names'].get(new_id, class_name)
            else:
                new_class_names_by_id[class_id] = replace_yaml['new_names'].get(class_id, class_name)
        
        new_class_names_by_id = dict(sorted(new_class_names_by_id.items()))
        #print(new_class_names_by_id)
        with open('new_class_names_by_id.yaml','w') as f:
            yaml.dump(new_class_names_by_id, f)

    class_id_replace = {str(key): str(value) for key, value in class_id_replace.items()}
    #print(class_id_replace)
    #exit(1)

    for lb_file in file_list:
        label_filename = new_lb_folder.join(lb_file.rsplit(old_labels,1))
        new_label_dir = Path(PurePosixPath(label_filename).parent)
        new_label_dir.mkdir(parents=True, exist_ok=True)
        with open(str(lb_file)) as f:
            new_annotation = ""
            for x in f.read().strip().splitlines():
                if len(x):
                    lb = x.split()
                    class_id = lb[0]
                    if class_id in class_id_replace:
                        new_class_id = class_id_replace[class_id]
                    else:
                        new_class_id = class_id
                    lb[0] = new_class_id
                    if new_class_id != "-1":
                        new_annotation += ' '.join(lb) + '\n'
            if len(new_annotation):
                with open(label_filename, "wt") as fout:
                    fout.write(new_annotation)
