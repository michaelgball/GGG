library(tidyverse)
library(tidymodels)
library(vroom)

trainSet <- vroom("train.csv")%>%
  select(-id)
testSet <- vroom("test.csv")

trainSet$type <- as.factor(trainSet$type)

my_recipe <- recipe(type~., data=trainSet) %>%
  step_mutate(color <- as.factor(color)) %>%
  step_dummy(all_nominal_predictors()) %>%
  step_normalize(all_numeric_predictors()) %>%
  step_pca(all_predictors(), threshold=.9)

  #step_impute_knn(missSet$bone_length, impute_with = all_numeric_predictors(), neighbors = 10) %>%
  #step_impute_knn(missSet$hair_length, impute_with = all_numeric_predictors(), neighbors = 10) %>%
  #step_impute_knn(missSet$rotting_flesh, impute_with = all_numeric_predictors(), neighbors = 10)

library(discrim)
library(naivebayes)

## nb model
nb_model <- naive_Bayes(Laplace=tune(), smoothness=tune()) %>%
  set_mode("classification") %>%
  set_engine("naivebayes") 

nb_wf <- workflow() %>%
  add_recipe(my_recipe) %>%
  add_model(nb_model) 

## Tune smoothness and Laplace here
tuning_grid <- grid_regular(Laplace(),smoothness(),levels = 4)
## Split data for CV
folds <- vfold_cv(trainSet, v = 5, repeats=1)

## Run the CV
CV_results <- nb_wf %>%
  tune_grid(resamples=folds,grid=tuning_grid,metrics=metric_set(accuracy)) 

## Find Best Tuning Parameters
bestTune <- CV_results %>%
  select_best("accuracy")

final_wf <-nb_wf %>%
  finalize_workflow(bestTune) %>%
  fit(data=trainSet)

## Predict
nb_preds <- predict(final_wf, new_data=testSet, type="class") %>%
  bind_cols(., testSet) %>%
  select(id, .pred_class) %>%
  rename(type=.pred_class)

vroom_write(x=nb_preds, file="./NB_Preds.csv", delim=",") 


##Neural Networks
library(nnet)
trainSet <- vroom("train.csv")
trainSet$type <- as.factor(trainSet$type)

nn_recipe <- recipe(formula=type~., data=trainSet) %>%
  update_role(id, new_role="id") %>%
  step_mutate(color <- as.factor(color)) %>%
  step_dummy(all_nominal_predictors()) %>%
  step_range(all_numeric_predictors(), min=0, max=1) #scale to [0,1]

nn_model <- mlp(hidden_units = tune(),epochs = 50) %>%
  set_engine("nnet") %>% 
  set_mode("classification")

nn_wf <- workflow() %>%
  add_recipe(nn_recipe) %>%
  add_model(nn_model) 

nn_tuneGrid <- grid_regular(hidden_units(range=c(1, 10)),levels=10)
folds <- vfold_cv(trainSet, v = 5, repeats=1)
tuned_nn <- nn_wf %>%
tune_grid(resamples=folds,grid=nn_tuneGrid,metrics=metric_set(accuracy))

tuned_nn %>% collect_metrics() %>%
filter(.metric=="accuracy") %>%
ggplot(aes(x=hidden_units, y=mean)) + geom_line()

bestTune <- tuned_nn %>%
  select_best("accuracy")

final_wf <-nn_wf %>%
  finalize_workflow(bestTune) %>%
  fit(data=trainSet)

## Predict
nn_preds <- predict(final_wf, new_data=testSet, type="class") %>%
  bind_cols(., testSet) %>%
  select(id, .pred_class) %>%
  rename(type=.pred_class)

vroom_write(x=nn_preds, file="./NN_Preds.csv", delim=",") 

##Boosting
library(bonsai)
library(lightgbm)
my_recipe <- recipe(formula=type~., data=trainSet) %>%
  update_role(id, new_role="id") %>%
  step_mutate(color <- as.factor(color)) %>%
  step_dummy(all_nominal_predictors()) %>%
  step_normalize(all_numeric_predictors())

boost_model <- boost_tree(tree_depth=tune(),
                          trees=tune(),
                          learn_rate=tune()) %>%
set_engine("lightgbm") %>% #or "xgboost" but lightgbm is faster
  set_mode("classification")

boost_wf <- workflow()%>%
  add_recipe(my_recipe) %>%
  add_model(boost_model)

boost_tuneGrid <- grid_regular(tree_depth(),trees(),learn_rate(),levels=3)
folds <- vfold_cv(trainSet, v = 5, repeats=1)
tuned_boost <- boost_wf %>%
  tune_grid(resamples=folds,grid=boost_tuneGrid,metrics=metric_set(accuracy))

bestTune <- tuned_boost %>%
  select_best("accuracy")

final_wf <-boost_wf %>%
  finalize_workflow(bestTune) %>%
  fit(data=trainSet)

## Predict
boost_preds <- predict(final_wf, new_data=testSet, type="class") %>%
  bind_cols(., testSet) %>%
  select(id, .pred_class) %>%
  rename(type=.pred_class)

vroom_write(x=boost_preds, file="./Boost_Preds.csv", delim=",")


##BART
bart_model <- parsnip::bart(trees=tune()) %>% # BART figures out depth and learn_rate
  set_engine("dbarts") %>% # might need to install
  set_mode("classification")

bart_wf <- workflow()%>%
  add_recipe(my_recipe) %>%
  add_model(bart_model)

bart_tuneGrid <- grid_regular(trees(),levels=3)
folds <- vfold_cv(trainSet, v = 5, repeats=1)
tuned_bart <- bart_wf %>%
  tune_grid(resamples=folds,grid=bart_tuneGrid,metrics=metric_set(accuracy))

bestTune <- tuned_bart %>%
  select_best("accuracy")

final_wf <-bart_wf %>%
  finalize_workflow(bestTune) %>%
  fit(data=trainSet)

## Predict
bart_preds <- predict(final_wf, new_data=testSet, type="class") %>%
  bind_cols(., testSet) %>%
  select(id, .pred_class) %>%
  rename(type=.pred_class)

vroom_write(x=bart_preds, file="./Bart_Preds.csv", delim=",")
