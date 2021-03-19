This is a project for image classification that was available on kaggle.

The problem here is to detect potential tubes in various positions (endotracheal, nasogastric, central venous and/or if the Swan Ganz catheter is present) and if they have a good placement (normal / borderline / abnormal). It is a multilabel classification problem.

The training dataset that contains the training target is the "train.csv". The images that are used for classification are in the train directory.

The train.csv have the following columns (completed): 

	StudyInstanceUID - unique ID for each image
	ETT - Abnormal - endotracheal tube placement abnormal
	ETT - Borderline - endotracheal tube placement borderline abnormal
	ETT - Normal - endotracheal tube placement normal
	NGT - Abnormal - nasogastric tube placement abnormal
	NGT - Borderline - nasogastric tube placement borderline abnormal
	NGT - Incompletely Imaged - nasogastric tube placement inconclusive due to imaging
	NGT - Normal - nasogastric tube placement borderline normal
	CVC - Abnormal - central venous catheter placement abnormal
	CVC - Borderline - central venous catheter placement borderline abnormal
	CVC - Normal - central venous catheter placement normal
	Swan Ganz Catheter Present
	PatientID - unique ID for each patient in the dataset

with the associates image IDs and patient IDs.


The test directory contains the image of the patients for whom we have to predict the catheter presence and position.


sample_submission.csv is an example of what the script should produce.


algogo.py is the python script use to train the model and create a dataset from the train images predictions.


efficientnetb3_notop.h5 is the model used for the classification (EfficientNet B3)