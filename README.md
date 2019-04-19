# PatternRecognition_JungleSpeed
The app folder contains the models we used for the different tasks

The report folder contains the reports for the exercices, one report per task with the results in it (% accuracy, loss...)

The results folder contains the results of each task

## Keyword spotting task
Create a profile under data/ks/profile (have a look at base.cfg).

Then you can call the keyword spotting app `ks.py` with your profile as parameter, e.g. for running the app with the cluster config you can call it as follow `python3.7 ks.py cluster`
 
###how to configure
####main 
```buildoutcfg
[main]
name=base
log_to_file=1
```
@main is the config name, currently not realy used
@log_to_file if 1, the logging output is directed to result/ks/{datetime}/main.log

####jobs
```buildoutcfg
[jobs]
items=crop,featureSelection,train,validate

```
@items you can define the jobs in the corresponding order where they should be executed you need to use the config names of the jobs
####job
every job is prefixed by `job_` followed by his name/key. If you don't define the property `type`, the fallback job `ImagePreProcessing` is taken
````buildoutcfg
[job_crop]
input_directory=../data/ks/image/
function=crop
svg_dir=../data/ks/ground-truth/locations/
apply_polygon_mask=1
````
For the image `ImagePreProcessing` you can choose the pre processing function by setting the `function` property
For the `crop` function you need to define some additional properties.

@svg_dir path to the svg locations with the polygons

@apply_polygon_mask if 1, all pixels out of the polygon are set to the mean background color define by the yen threshold









