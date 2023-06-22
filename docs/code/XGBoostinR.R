library("mlr")
library("mlrMBO")
library("magrittr")
library("emoa")

#This is the model fitting script.  We'll be using XGBoost with several regularization hyperparameters tuned.  The algorithm will be nested within several other algorithms that support it.  The outermost of these support algorithms will be a feature selection wrapper using a filtering feature importance index that has been modified to reweight importance in a manner that penalizes more vs less missingness.  The feature selection wrapper first does mean imputation on the training data feature set, then computes a simple Mutual Information index. With this modification, the imputed means are discarded before passing the data in towards the next layer, and therefore the final model parameters aren't based on any imputed values.  The next layer is a form of Bayesian optimization deployed to estimate the optimal hyperparameters. This inner algorithm is a Krigging algorithm, itself built with the aide of an evolutionary search strategy.  Rather than fit an XGBoost model for every set of hyperparameters, the Krigging algorithm takes a small set of XGBoost models fit with a set of hyperparameters to attempt to predict how an XGBoost model should perform given new hyperparameters. It serves the funciton of identifying areas of the hyperparameter search space that exhibit both high uncertainty with regard to performance and high expected improvment. The evolutionary search strategy, in this case with 10000 points for each generation, assists the Kriging algorithm by evaluating candidate hyperparameter sets for expected improvement. Given the observed performance on other hyperparameters sets and expected improvement, a new candidate solution, is evaluated and the result is added to pool iteratively until the number of iterations,in this case 2000, is reached. After the cycle at 2000, 10 final models are fit and averaged to smooth out noise and pick the final, optimal, hyperparameters. 

#mlrMBO vignette: https://mran.microsoft.com/snapshot/2017-03-22/web/packages/mlrMBO/vignettes/mlrMBO.html
#FSelectorRcpp: https://mi2-warsaw.github.io/FSelectorRcpp/articles/get_started.html 
filters = as.list(mlr:::.FilterRegister)
filters$FSelectorRcpp_gain.ratio$fun

mlr::makeFilter(
  name = "FSelectorRcpp_gain.ratio.with.mean.imputation",
  desc = "Does mean imputation then computes an index of mutual information ",
  pkg = character(0),
  supported.tasks = c("regr"),
  supported.features = c("numeric", "integer","double"),
  fun = function (task, nselect,decreasing=TRUE, ...) 
  {
    data = getTaskData(task)
    X = data[getTaskFeatureNames(task)]
    y = data[[getTaskTargetNames(task)]]
    X = apply(X,2,function(x){replace(x,is.na(x),mean(x,na.rm=T))})
    res = FSelectorRcpp::information_gain(x = X, y = y,equal=TRUE, type = c("infogain"), 
                                          ...)
    res = setNames(res$importance, res$attributes)
    replace(res, is.nan(res), 0)
  })
####Can compare the non-imputed selected features with the mean imputed version: 
## generateFilterValuesData(task,"FSelectorRcpp_information.gain")
# generateFilterValuesData(task,"FSelectorRcpp_gain.ratio.with.mean.imputation") 
# A note on the rationale: Feature seleciton is generally developed with and intended for use on data with little to no missingness.  In practice this is essentially never the case.  We can impute values in a variety of ways to close the gap, for example imputation via prediction itself but these are exponentially more computationally intenseive. Mean imputation however isn't.  Given a substantial degree of missing information in a vector, even if a feature is truly predictve we won't be able to predict with it. So, heuristics win the day.  Penalize the estimates.  But what if we have a truly predictive feature that we penalize too much?  See 'fw.perc' below

#read in the dataset.
df <- read.csv(file.choose()) 
#If a feature has variance equal to 0, remove it. 
df <- mlr::removeConstantFeatures(df,perc=0) 

#Our no information rate  is rmse. Instead of using a 'naive learner' or a 'featureless learner' we're just going to use the sample standard deviation as our no informaiton rate.  If our model doesn't produce an error that is smaller than this error rate, then it means that our model 'does not reduce uncertainty' 
no.info.rate <- mean((df$target-mean(df$target))^2) 
table(apply(df,2,class))
#Note that R-squared is 0 ->  1 - no.info.rate/no.info.rate 
#This formula is different from the R-squared formula used in R functions like lm().  lm() uses sum((pred-y - mean(y))^2)/sum((true-y - mean(y))^2) and could give us a value of .999 even the model is terrible....has to do with the intercept in the equation See:Kvalseth,T.O., (1985), Cautionary Note about R2.The American Statistician, Vol. 39, No. 4, Part 1, pp. pp. 279-285 

#Assuming your data object is imported and dependent variable is named 'target', you can just run the next line and the whole script will execute. You don't have to use the loop but it makes rerunning the code faster. 
for(execute in 1){
  #If you are doing a classification problem, then this is not the correct formulation.  Instead you could use:
  #odelMetrics::logLoss(actual=as.numeric(df$target),rep(mean(as.numeric(df$target)),nrow(df)))))
  no.info.rate <- mean((df$target-mean(df$target))^2) 
  
  lrn <- makeFilterWrapper( 
    learner = makePreprocWrapperCaret(
      # makeImputeWrapper( # included for completeness. This wrapper could be deployed for imputing missing values via prediction.  The imputation would occur within the cross-validation scheme
      mlr::makeLearner(
        cl = "regr.xgboost", 
        predict.type = "response", 
        par.vals = list(objective = "reg:squarederror",base_score=mean(df$target),eval_metric = "rmse"),#Set the base_score to the mean of the target vector.  In a vanilla gradient boost algorithm, the algorithm would do this but in XGBoost the code is more flexible.  Sometimes it might make sense to pick another base_score depending on your distribution.  In our case, we're using the idea that in a standard linear mulitple regression, when all of the predictors have Betas of 0 (meaning no predictive contributions), the line of best fit will go through Y= intercept...which will probably be the mean of Y.  This has the added benefit of saving around 25-30% of our computation time. 
        verbose = 0, 
        config = list(on.par.without.desc = "warn")),#, 
      #classes = list(numeric=imputeMean(),integer=imputeMean())),imputeLearner("regr.randomForestSRC") #in .randomForestSRC vs .randomForest the impute option is set to True to allow for missing values #This code is included to illustrate where and how imputation could be done.  We could impute means internally or we could use another algorithm, for example random forest to impute via prediction. 
      ppc.scale=TRUE), #We're going to preprocess our data before building the models It isn't strictly necessary for XGBoost but if you added custom regularization terms this would be how to get the data standardized in this framework. This is just z-scoring the data. This is important if variables have different scales as their associated weights will be scale dependent.  Think unstandardized beta weight in regression for example.   regularzation parameters tend to apply to all weights equally so if scales for different variables are very different, the optimization algorithm will select a regularization penalty at random because performance will be distorted with each level of the parameter.  The result will tend to be most optimal for values that are much larger than would otherwise be the case and the final model will perform poorly if at all better than the null model. In the case of XGBoost, however, what's being regularized is the quality of splits based on information gains. In other words, the scale is just the scale of the error function and it doesn't matter how the indepdnent variables are scaled. However, scaling the dependent variable can help with setting appropraite values for hyperparameters as I explain in the Python script. 
    fw.method = "FSelectorRcpp_gain.ratio")
  #The custom feature filtering index.  Mutual information is used for that index.  Mutual information can be thought of as a form of non-linear correlation.  In other words, the universe of candidate solutions isn't restricted to only linear relationships.  This particular implementation uses discretization and binning to acheive the estimate and it is known that discretizing and binning are not ideal.  However, I've also noted on similar other problems that feature selection with Mutual information rather than pearson or spearman correlation produced, ultimately, better, in terms of higher R-squared/lower error, models on the same data.  
  
  #### Open:  This section creates the hyperparameter search space ###
  #The XGBoost website: https://xgboost.readthedocs.io/en/latest/parameter.html  
  
  par.set <- makeParamSet(
    # makeNumericParam("tweedie_variance_power",lower = 1,upper = 2),
    makeNumericParam("fw.perc", lower = 1/(ncol(df)-1), upper = 1), # How many features to keep after computing the feature selection index.  Rather than determine manually how many features to retain, we tune the decision just like every other hyperparameter.  In this way, if the feature seleciton index errantly indicates that some set of truly non-predictive features is supposed to be really predictive, the algorithm will suggest to retain larger and larger numbers of total features so that it can use features that actually are more predicitive but were too heavily penalized from the mean imputation above.  We examine these values after the model has been fit to see whether or not such a thing occurred.  When a strong 'signal' is present in the data we'll see that the algorithm quickly hones in on a narrow range of candidates.  This entire process is conducted within the cross-validation scheme. This means that the total number of features and the specific features themselves can be loosely interpreted as a set of estimators that are consistently well related to the target regardless of sample compositiion. #NOTE: The scope of the optimization problem determines the boundaries for this hyperparameter.  If there are many thousands of features, the problem is very complex and the optimization will likely be suboptimal if using the full range.  For example 700 datapoints with 3000 variables would be a problem if the range was from 1 feature to all of them.  In these cases, optimize the problem first and then examine the values visited by the optimization.  It should opt for a small number of features but it may not. In that case, restrict the range manually and run the optimzation again.  So, search from 1 to 10% of the features.  This will force the optimizer to propose points all within a range that has a better chance of performing well.
    
    
    makeNumericParam("nrounds", lower = 0, upper = 5,trafo = function(x){round(10^x,digits=0)}),
    #Tranformation faciliates searching the hyperparameter space faster. There are different options for this.  Another common option is log-uniform. In the present case, the optimization algorithm picks values from 0 to 5 instead of from 1 to 100000.  If this doesn't work well in the first round it can be disabled. 
    #how many trees to consider building. More trees equals more time processing.When we expect there to be little to no predictive power from our feature set given the fact that our previous study found little to no predictive power, we'll use a large number of trees to extract as much potential from the data as we can computationally handle. Note that the number of trees needed is not simply a function of the complexity in the data but also of the strength of the regularization penalty terms.So more trees doesn't actually mean more complexity in all cases. XGBoost builds trees by reducing the residuals from the previous tree and then pruning away branches if they don't meet the criterias set by the regularization. Parameters like eta and lambda 'shrink' the reduction in residuals from a given tree, thus leaving 'more' residual error for the next tree to work with.  This allows the algorithm to more gradually/slowly approach an optimal solution without overfitting. When overfitting occurs, the training errors are overoptimistic and the test errors are worse. In the case where there is expected to be little to no real signal, we have much less tolerance for overfitting so the regularization will need to be strong to ensure a slow approach.   
    makeIntegerParam("max_depth", lower = 1, upper = 120),    # Maximal number of levels in any tree. #This is not the same thing as an interaction. Splits can occur repeatedly along a single feature and even in the case where there are several features involved, the interaction is far more similar to a 'piecewise function' than the product of two vectors.  This hyperparameter merely determines how many splits are allowed to occur within any given tree, itself still restrained by the regularization criteria. In particular, gamma directly influences tree complexity. Lambda, and alpha do as well but in a different way. 
    makeIntegerParam("early_stopping_rounds", lower = 1, upper = 250),#stop building trees if no progess after this many trees have been built.  VERY useful to use this hyperparameter.  The tree building process in XGBoost is defined such that the algorithm will simply build as many trees as we have specified.  Technically, tuning only the number of trees to build can get us good performance.  But tuning early stopping rounds as well, gives the optimzation algorithm more information to use in gravitating in on an optimal set of values and should push the final solution in the direction of a simpler/smaller model.    
    
    makeNumericParam("eta", lower = -10, upper = 0,trafo= function(x){2^x}),   # "shrinkage" or 'Learning rate' - prevents overfitting. This hyperparameter  has to do with how quickly the algorithm moves 'down' the gradients.  Large numbers means the algorithm takes larger 'steps' and that means it could step right over the 'sweet spots' where there are 'pockets' of lower error. This hyperparameter works by taking the final 'reduction in error' for the tree being currently built and multiplying it by eta to put a downward pressure on the amount of contribution any single tree can have.  
    makeNumericParam("gamma", lower = -10, upper = 10,trafo= function(x){2^x}),  #regularization parameter. #Gamma is multiplied by the number of Leaves on a Tree and added to .5*lambda*w^2 in the objective function. The objective function contains the loss associated with predicted values (i.e., residual error). Adding in Gamma and the influence of Lambda will shrink the loss, the providing less 'space' for further optimization. You can image it as artifically making the predictions more accurate to prevent the algorithm from learning more. Basically, gamma controls the number of leaves in the trees by setting a threshold for the reduction in loss required for a split to be retained. Splits that don't meet the requirements of gamma are pruned away. Therefore, it regulate the maximum depth. 
    makeNumericParam("subsample", lower = .01, upper = 1),
    #If the data are noisy or there are outliers present, this parameter randomly subsamples the number of cases to consider during the tree building process. 
    makeNumericParam("lambda",lower = -10,upper = 10, trafo = function(x) {2^x}), 
    # reduces the sensitivity of the algorithm to individual points.  Is added to the denominator of the similarity scores which, in turn, makes the gain scores lower and increases the chance that a particular feature will have a large contribution to the final model...unless the estimates are consistently meeting these threshold criteria. Put differently, similarity scores are used when constructing the tree. Lambda, therefore, influence how the tree is constucted. But it also shows up in defining what the weight will be for a given leaf. In particular it 'shrinks' the weight by appearing in the denominaotr.  And lastly, as a consquence, it also shows up in the objective function as described above with gamma.  Has the function similar to L2 norm, shrinks weights but not often to 0.
    makeNumericParam("alpha",lower = -10,upper = 10, trafo = function(x) {2^x}), #L1 norm, shrinks feature weights to 0.
    #shows up only in the objective function.  
    makeNumericParam("colsample_bytree", lower = .01, upper = 1))
  #Subsamples the number of features to consider for any given tree. Features are competing for space in the tree at all stages.  Sometimes the data on hand systematically prevent a potentially useful other feature from playing a role in the model.  This parameter allows for those other features to appear in the final model and in some cases if a feature turns out to be a useful predictor but only when conditioned on another predictor, for such relationships to be more effectively modeled. 
  
  des = generateDesign(n=300,par.set = par.set,fun = lhs::maximinLHS) #This is maxmin latin hypercube sampling 

  #### Close:  This section creates the hyperparameter search space ###
  
  
  ### Open: This section prepares the Rsession for parallel computing ###
  configureMlr(on.learner.warning = "quiet", show.learner.output = FALSE) #prevent output from flooding your console. Less output on your console allows your system to use those resources for driving the computation
  
  parallelMap::parallelStart("socket",parallel::detectCores(),level = "mlr.resample",show.info = FALSE) #IF running on Mac you can use 'multicore' but if on Windows you can use 'socket' Mac can use socket too but multicore does a 'FORK'  This can save a lot of RAM and compute time.
  
  ### Close: This section prepares the Rsession for parallel computing ###
  
  
  Start.time <- Sys.time()
  print(Start.time)
  
  #### Open:  This is the algorithm.  Subsections are outlined  ###
  
  ######## Open: mlrMBO ###
  # https://cran.r-project.org/web/packages/mlrMBO/vignettes/mlrMBO.html
  #https://github.com/mlr-org/mlrMBO
  #MBO goal.  Find the best hyperparameters with the least amount of resource expenditure. Random search discards valuable information from every fit, yet every model fit is expensive. MBO stores information from every model fit and uses that information to select the next best candidate set of hyperparameters.
  ###Curse of dimensionality.  IF we have 2 hyperparameters and we want to evaluate just 10 levels each, we'll have to fit our model 100 times to exhaustively search the set.... IN a 10 fold cross validation scheme that means we to have to fit 100 * 10 models and we're still only using 10 values for each of the hyperparameters. In the case of XGBoost, many hyperparameters are continuous and unbounded on the upper end.  So to do a thorough search we would want to test far more values than just 10.  Moreover, we'd want to use far more than just 2 hyperparameters as seen above.  If we tuned just 4 then in the previous example we'd have to fit 10^5 = 100,000 models to test just 10 values of each hyperparameter.  In this script, we'll test a few hundred levels of each hyperparameter, with 8 hyperparameters, in a 5*5 resample scheme.  That'd be ballpark 100 values to the power of 8 = 1e+16... times 25... but all of that will be achieved by substituting the expensive model with a cheap model that attempts to predict how the expensive model will behave and uses its predictions to internally 'experiment.' The effect of this is that many of these combinations need not actually be tested because they have a higher probability of poorer performance, given the performance of adjacent points and given the assumptions that the search space is relatively smooth with regard to performance (which tends to be true given that regularization effectively 'smooths' functions and tends to increase predictive performance) . 
  
  #MBO inner tuning controls.
  mlrMBO::makeMBOControl(
    # save.on.disk.at = c(500,1000,1500,2000), #Algorithms crash for any number of reasons.  XGBoost is pretty stable but if your system runs out of RAM and becomes unstable that could be a problem.  This chunk saves a copy of the state of your MBO search at these iterations.  The xgboost model itself will need to be re-estimated but these checkpoints provide the information hyperparameters and past evaluations.
  
    store.model.at =c(401), #Since we don't actually know what the 'optimal' model will be, this indicates the final point at which the model would be stored in the object to be returned should occur at the end of the search.  

    resample.at = c(0:25), #This does resampling on the candidate solutions at each of the specified iterations. Iteration 0 is for the 'initial design' is a hypercube which can be thought of as an evenly distributed set of candidate solutions across the entire hyperparameter search space.The idea here is to set a trajectory that is very reliable for the search effort. Even though later candidate solutions may be a bit more noisy, because we've set this solid foundation these other solutions should still be good. It's another tradeoff between efficiency and precision.  In principle, you can resample at every iteration. Doing so will incur more compute time though.
    resample.desc = mlr::makeResampleDesc(method = "CV",iters=10, stratify = FALSE), # This is for cross-validating the surrogate model.  If you plan on tuning the surrogate model then you would use this.  Otherwise, it can be good for determining how well your surrogate model learned the relationship between hyperparamters and model performance.  If neither of those are applicable to you, then this should be skipped/commented out along with the resample.at line above it and the next line. then 'best.predicted' should be changed to 'best.true.y'
    resample.measures = list(setAggregation(mse,testgroup.mean)), #with repeated cv aggregate first within each round and then take the mean of those means. 
    final.method = "best.predicted",
    final.evals=10) %>% #best.predicted means that at the end the surrogate model (the cheap model) revisits all proposed points and attempts to predict them again.  The hyperparameters for the best predicted point is returned.  This is supposed to help when the signal is weaker by smoothing over some of the variation from prior predictions. Final evals is how many times the process is repeated on the chosen point. Again, none of that is needed if you aren't adjusting the surrogate learner parameters. So, you can comment out that stuff and pass line 119 forward by deleting the, and adding a %>% pipe. 
    
    # Syntax note about the %>% pipe:  From here onwards we're using the magrittr pipe '%>%' operator and a '.' period.  The pipe operator says to take the thing on the left and insert it into the .  on the right. 
    mlrMBO::setMBOControlTermination( .,iters =100) %>%    #This is how many 'steps' our search algorithm will ultimately take. It is our 'degree of search' parameter. #Different stopping criteria can be used with mlrMBO.  We can use various time budgets or execution budgets or even a target precision threshold.  This criteria is just saying to test 100 candidate solutions after completing the initial design evaluations.
    
    mlrMBO::setMBOControlInfill(.,    #There are different 'cheap' model updating strategies.  Here the search is guided by an evolutionary algorithm.  
                                crit = mlrMBO::makeMBOInfillCritEI(), #EI is 'expected improvement' 
                                #opt.focussearch.points = 5000, for opt = "focus search"  which is the default.  This is less memory intensive than the evolutionary search and a good alternative. 
                                opt.restarts = 3, #Helps to get around crashes/local optimums. 
                                opt = "cmaes",  #Evolutionary algorithm
                                opt.cmaes.control = list(lambda=500,stop.on=stopOnMaxIters(12000))) %>%   #The number of points in each generation
    # here's a great visulaization of what the evolutionary search strategy is doing. #http://blog.otoro.net/2017/10/29/visual-evolution-strategies/
    mlr:::makeTuneControlMBO(
      mbo.design = des, #path is a set of points that have already been evaluated.   see: 'initial design' https://cran.r-project.org/web/packages/mlrMBO/vignettes/mlrMBO.html ###In this script path is created by first running the algorithm in its entirety.  Then the of optimization that was used is extracted and can be plugged in here so that when the algorithm is run a second time, it won't revisit points that it has already evaluated. Note:  this influences the search behavior of the algorithm.  If there is an unexplored part of the space that the algorithm seems to be ignoring, this won't help.  To explore that space, a separate feature "proposePoints" will need to be used.  *Not sure if this can be done as part of the initial evaluation or if this can be done after the final object is constructed but before starting a second pass with the path included in the design.
      learner = mlr::makeLearner("regr.km", predict.type = "se",nugget=10^(-8)*1,scaling=TRUE,covtype='gauss'),#var(path$y,na.rm = T)),# if des is not null at y; nugget=10^(-8)*1 if y is null
      #covtype = gauss refers to the Squared Expontential (a.k.a. RBF kernel).  
      #DiceKrigging.  This is the cheap model itself.  There are other options but this one is apparently very good. For example, you should be able to use a random forest here if you wanted.  #Explanation for nugget: https://github.com/mlr-org/mlrMBO/issues/80 ##nugget is needed if mbo.design from previous iteration is in use... 
      mbo.control = .) %>%
    
    ######## Close: mlrMBO ###
    
    mlr::tuneParams(  #Outer resampling loop.
      # mlr::makeTuneWrapper( if using featselwrapper feature selection through wrappers is computationally expensive but this is where that would go in the code.
      learner= lrn,
      resampling = mlr::makeResampleDesc(method = "CV",iters=10,stratify = FALSE),
      task= mlr::makeRegrTask(id="Name your model if you want",data = df ,target = "target"),
      measures = list(setAggregation(mse,testgroup.mean)),#,setAggregation(rmse,testgroup.sd),rmse,rsq,setAggregation(rsq,testgroup.mean),timetrain), #These other measures can be computed as well. Only the leading measure will be optimized though.   You can also just run an additional cross-validation routine to compute additional metrics once you have selected optimal hyperparameters. XGBoost models only take a few moments to fit. 
      par.set = par.set,
      control = .,
      show.info = FALSE)  -> xgboost.with.regularizingparameters
  
  #### Close:  This is the algorithm.  ###
  
  # ##### Open:  Model evaluation ###
  #This is the estimate of our error.  Is it lower than the no.info.rate
  #no.info.rate #RMSE from naive prediction
  #xgboost.with.regularizingparameters$y #RMSE from fit.
  
  #Estimated R-squared:  #Unbounded at 0
  print(1-xgboost.with.regularizingparameters$y/no.info.rate)
  
  End.time <- Sys.time()
  print(End.time-Start.time)
  
  parallelMap::parallelStop()  # stop the parallel processing to reclaim system resources.  If using 'PSOCK' this is essential.  
} 