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
