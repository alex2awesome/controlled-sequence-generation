################################################################
####################### TRAIN PREDICTORS #######################
################################################################

# Train RACE Predictor
allennlp train src/predictors/race/race_roberta.json \
	--include-package src.predictors.race.race_dataset_reader \
	-s trained_predictors/models/race/

# Train IMDB Predictor
allennlp train src/predictors/imdb/imdb_roberta.json \
	--include-package src.predictors.imdb.imdb_dataset_reader \
	-s trained_predictors/models/imdb/

# Train Newsgroups Predictor
allennlp train src/predictors/newsgroups/newsgroups_roberta.json \
	--include-package src.predictors.newsgroups.newsgroups_dataset_reader \
	-s trained_predictors/models/newsgroups/



allennlp train editing/src/predictors/news_discourse/news_discourse_roberta.json \
	--include-package editing.src.predictors.news_discourse.news_discourse_dataset_reader \
	-s editing/trained_predictors/models/news_discourse/

################################################################
########################## STAGE ONE ###########################
################################################################

STAGE1EXP=mice_gold

python run_stage_one.py -task imdb -stage1_exp ${STAGE1EXP} 
python run_stage_one.py -task newsgroups -stage1_exp ${STAGE1EXP} 
python run_stage_one.py -task race -stage1_exp ${STAGE1EXP}