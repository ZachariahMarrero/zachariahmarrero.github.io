#%%
'''This script is designed to optimze the hyperparameters of the XGBoost algorihtm.  It does so with the help of a few tools. 
1. It uses a custom version of cross-validation which we might informally refer to as stratified cross-validation.  
2. The cross-validation routine is manually defined and asynchronous.  With a manual definition, the code can be extended easily. With asynchronous processing, different combinations of hyperparamters can be tested in parallel (i.e., CPU cores will not be sitting idle while waiting for other processes to terminate).
3. The hyperparamters will be tuned with Bayesian Optimiztaion using Gaussian Process Regression. 
The gaussian process is built using GPyTorch. GPyTorch is unique amongst gaussian processing libraries in that it is build to be both customizable (as you will see) and highly scalable (up to billions of datapoints).
4. In addition to being scalable, GPyTorch also implements state-of-the-art Kernels.  In this script the Kernel of choice is the Spectral Mixture Kernel (read about it here: https://arxiv.org/pdf/1302.4245.pdf). But it the library can also go even further with Deep Kernel Learning. 
6. In initiating the GPyTorch model, this script will use Maxmin Later Hypercube Sampling.  In short, this is an approach to randomly sampling from a search space that ensures the sampled points cover the entire search space well. If we simply used random search, there would be large pockets of unrepresented space. Later Hypercube Sampling provides a strong starting condition for hyperparamter tuning and will help design the matrix upon which the GPyTorch model will be built. 

7. To guide the bayesian optimization routine, this script will BoTorch.  BoTorch is built on top of GPyTorch.  It too has state-of-the-art features.  In this script, a key feature in BoTorch that will be used is the one-shot knowledge gradient acquisition function.  '''

#Step 1, import our models and functions.
# # If you don't hav ea module, you can install it with:  !pip install name 
import concurrent.futures # For multiprocessing
import numpy as np #For data handling
from pyDOE2 import lhs #For latin hypercube sampling
import random #For random sampling
from scipy.stats import beta #For adding a prior to the latin hypercube samples
from sklearn.model_selection import KFold # For defining cross-validation folds
from sklearn.pipeline import Pipeline #For adding preprocessing to a model pipeline
from sklearn.preprocessing import StandardScaler #For staling model features
import torch # For Guassian process setup. #Before installing, make sure you have CUDA libraries installed correctly. 
import xgboost as xgb #XGBoost algorithm 
import gc #Garbage collection for memory management
import pickle #For saving and loading files 



#%%
#Step 2, load the data
#This is not strictly needed but for simple data analysis it helps.  This function ensures that 
#file_path = askopenfilename() # The R equivalent here is filename = file.path(file.choose())
filename = r"C:\Users\Zachariah\Desktop\data.csv"

#Instead of loading to a dataframe, we're loading to a numpy array. an array is just a matrix that can have more than 2 dimensions. 
# First, load the column names/ headers
headers = np.genfromtxt(filename, delimiter=',', dtype=str, max_rows=1)
headers = [str(x) for x in headers.tolist()]
#Second, Load the data.
# This function will replace missing or invalid values with NaN
data = np.genfromtxt(filename, delimiter=',', skip_header=1, filling_values=np.nan)

#%%
#Step 3, Setting up Cross-validation 

# Number of cross-validation folds
num_folds = 10
# maximum number of instances per fold
fold_limit = round(len(data)/num_folds,0)

#Rather than doing a simple random assignment, this script will do something a little more intentional. When your dataset contains relatively few observations some parts of your sample space may be poorly represented. Consequently, a simple random assignment *can* result enitre parts of the sample space being only represented in either a training or a validation fold but not both.  So, you want to be sure that poorly represented parts of the space are distributed well across your dataset.  For example, consider a binary classification problem with heavily imbalanced classes.  You might have 200 total observations of which only 20 are in the positive class.  If so, you want to make sure that those 20 don't end up all within 1 fold.  Typically, stratification is done.  However, it is also the case that in regression problems similar issues arise.  


# initialize cv_fold_index and cv_data as empty lists
cv_fold_index = []
cv_data = []

# initialize fold_counts as a dictionary
fold_counts = {i: 0 for i in range(1, num_folds+1)}

#To stratify, you start by taking the lowest occuring values and distributing them first. 
#First, We want values that are close in the numerical space to be distributed well in the case of regression. So, we'll adopt a heuristic that joins together proximal values. In this case we'll use a histogram.  The histogram uses the Freedman-Diaconis rule to automatically determine the bin edges that define which values get clustered together.
hist, bins = np.histogram(data[:,0])
binned_values = np.digitize(data[:,0],bins)

#Second, get the unique elements in binned_values and their counts
unique_elements, counts = np.unique(binned_values, return_counts=True)

#Third, sort unique_elements by their counts in ascending order. The idea here is that we want to start with those lowest frequency values. Doing so allows us to generalize this procedure to multiple dependent variables (and possibly even to the independent variables). In this script, we're only focusing on the dependent variable though. 
sorted_unique_elements = [element for _, element in sorted(zip(counts, unique_elements))]

#Fourth, conduct the assignment. 
#Read as, for each unique element, where the element is the bin identifier from the histogram...
for i in sorted_unique_elements:
    index = np.where(binned_values == i)[0]
    #If the number of values in the bin is greater than 1, shuffle them to ensure we get a random assignment.
    if len(index) > 1:
        index = random.sample(index.tolist(), len(index))
        
    #Then, for each of those values
    for idx in index:
        # randomly assign a fold, ensuring that the fold is not already full ( in this case we're limiting the size of a training set to make sure they are all approximately equal. We don't technically have to do this but it makes aggregating our cross-validation estiamtes simpler later on.)
        while True:
            cv_fold = random.choice(range(1, num_folds+1))
            if fold_counts[cv_fold] < fold_limit:
                break
        fold_counts[cv_fold] += 1
        cv_fold_index.append(cv_fold)
     #  cv_data.append(idx) if we want to save the indexes for additional use or multiple sorting conditions.

#%%
#Fifth, we need to now sort our original data according to the ordering of the folds.  
data = data[np.argsort(cv_fold_index, kind='stable'),] #stable here means "if there are ties, do not randomly order them, instead sort them in the order in which they appeared.""  This is important! We assigned each observation above in a specific way and we are explicitly keeping that ordering when using stable.

# The next line defines our cross-validation scheme. We're saying that we will do num_folds and to not shuffle the data internally before taking the folds. Since we already allocated the data randomly above, we don't need a shuffle here. 
kf = KFold(n_splits=num_folds, shuffle=False)

#We no longer need these items, so they're getting removed. 
del [idx,i,index,cv_fold,fold_limit,counts,hist,binned_values,bins, unique_elements, sorted_unique_elements]
#%%
#Step 4, Defining our algorihtm.  We're using XGBoost but you could just as easily slot it something else here.  
#For XGBoost, we need to separate the independent variables (often just called 'the features') from the dependent variable (often just called 'the target')
#Says, use the headers object to identiy which column is labeled 'target'. Assign the target to y. Assign all other columns to x.
y = data[:, headers.index('"target"')]
X = data[:, [i for i in range(len(headers)) if headers[i] != '"target"']]

#Sidenote:  Because of some design choices that are coming up later, we need to save these objects in the same directory as this script.  In short, we're going to use parallel processing via the multiprocessing module. On Unix based systems, when you create parallel processes, the paraellel processes point back to the existing session. However, in Python on Windows, the parallel processes get a full copy of the memory from the host process from which they were launched.  In R, those parallel processes don't get anything. You actually have to export your objects to those processes. In any case, because we're launching these processes (called spawns), Python will want to duplicate the state of the sessions in each spawn. And, the problem can be that when it does so, it may execute code again that you don't want executed and possily wind up in an unintentional infinite regress. So, we'll make a separate script which will launch the multiprocessing in a subprocess. In that subprocess, only the relevant objects will get duplicated.  This is really important because if we launch the multiprocessing from this script we'll end up seizing a bunch of memory, especially from our GPU(s), and we won't be able to release it back to the operation system (i.e., your computer will freeze up and require a reboot. Also, Pro Tip: it's a smart idea to make sure you have saved before you launch any potentially heavy compute tasks).  
np.save("X.npy", X)
np.save("y.npy", y)
with open('kf.pkl', 'wb') as f:
    pickle.dump(kf, f)  #saved with pickle.  These are just saving the item as a binary object so that it can be reimported without any modifications.

#Next up, we're going to define the actual algorithm.  In this case, we're defining a custom estimator class for our XGBoost algorithm. You may be aware that Scikit learn already has an XGBoost algorithm. Despite that, we're going to use a custom estimator because Scikit learn's version may not necessarily contain all of the features we want to use that might appear in the most recent XGBoost module.  Similarly, if we want to customize the XGBoost source code ourselves this custom class will make that easier to integrate.  So, this will allow us to still benefit from Scikit learns interface without limiting our choices. 
# Define a custom estimator class for XGBoost

#%%
#You'll notice that all of this block is commented out.  It is commented out because we're not simply going to conduct a single cross-validation on one set of hyperparameters. Instead, we'll be testing hundreds of combinations of hyperparameters. 

#First, we define the hyperparameters we want to test like so: 
# xgb_params = {
   
#     'objective': 'reg:squarederror',
#     'num_boost_round': 100,
#     'booster':'gbtree',
#     'max_depth': 6,
#     'learning_rate': 0.1,
#     'base_score': np.mean(y),
    
# }

# #%%
#Second, we dfine the custom estimator which requires an initialization for attributes (__init__), a model fitting method, and a predict method. A method is just a function defined inside a class that can operate on an instance of the class.  So init, is defining the structure and fit()/predict() are defining what we can do with it. 
# class XGBEstimator:
#     def __init__(self, **kwargs):
#         self.params = kwargs
    
#     def fit(self, X, y):
#         dtrain = xgb.DMatrix(X, label=y)
        
#         self.model = xgb.train(params=self.params, dtrain=dtrain, num_boost_round = self.params.pop('num_boost_round'))
#         return self
    
#     def predict(self, X):
#         dtest = xgb.DMatrix(X)
#         return self.model.predict(dtest)

#Third, we define our processing pipline.  In this case we have StandardScaler to standardize the model feature. That's not needed for XGBoost because it doesn't influence the loss function.  However, you should consider scaling the dependent variable because the optimal values of hyperparameters such as gamma, alpha, and lambda are realtive to the magnitude of the gradients (i.e., residuals) and gradients have a scale that depends on the DV.  SO, by scaling the dependent variable, you can take away some of the guesswork in figuring out what values make sense for these hyperparameters. This is especially true when we define the bounds for our hyperparamter search.  What bounds make the most sense? By standardizing, we can know beforehand that the value we choose will be of comparable magnitude to the gradients, thus preventing the algorithm from ignoring them. For example, assume your dependent variable has a standard deviation of 1 million.  If you applied a gamma of 10, many researchers would tell you that a gamma of 10 is quite extreme (example: https://medium.com/data-design/xgboost-hi-im-gamma-what-can-i-do-for-you-and-the-tuning-of-regularization-a42ea17e6ab6), but in truth it might have no impact on the learning process at all in this case.
# pipeline = Pipeline([
#     ('scaler', StandardScaler()),
#     ('transform_y'),FunctionTransformer(func=StandardScaler().fit_transform(y.reshape(-1, 1)).flatten())
#     ('model', XGBEstimator(**xgb_params))
# ])


# Fourth, Compute the predictions for each fold using the pipeline if we don't want to save models
# #y_pred = cross_val_predict(pipeline, X, y, cv=kf, n_jobs=-1)

# #If we want to just do cross-validation and aggregate the results already, we can do that using cross_validate from scikit learn (remember to import the function first) as:
# cv_scores = cross_validate(pipeline, X, y, cv=kf, n_jobs=-1,scoring='neg_mean_squared_error',return_estimator=True) 
# #and compute the R-squared:
# mse_scores = -cv_scores['test_score']
# mse_mean = mse_scores.mean()
# print(1-(mse_mean/ np.mean((y - np.mean(y)) ** 2)))



#%%
#Step 5, Defining the 'design matrix' for our Gaussian process used in the Bayesian Optimization routine. 

#Before fitting a gaussian process, we'll want something for that model to start with.  Doing so will help to ensure our final results are more stable, absent any prior knowledge we could leverage to set our hyperparamters.  So, to get the process started, we will use a technique called latin-hypercube sampling.  The variant we're using is the maxmin, which does some additional process on the sampled points to attempt to maximize the minimal distance between the sampeled points.  In using this approach, we are sampling as uniformly as we're able to across the entire search space.

#First: Define the search space 'x': [min, max, is integer?]
xgb_params = {
    'num_boost_round': [1,50000,True],
    'max_depth': [1,25,True],
    'learning_rate': [2**-13,2**-1.7,False],
    'min_child_weight': [0,20,False],
    'subsample': [.01,1,False],
    'colsample_bytree': [.01,1,False],
    'colsample_bynode': [.01,1,False],
    'colsample_bylevel': [.01,1,False],
    'alpha': [2**-10,2**7.5, False],
    'lambda': [2**-10,2**7.5,False],
    'gamma': [2**-10,7,False]}

xgb_params_design = xgb_params
number_of_hypers_to_tune = len(xgb_params)
#I don't know of any rules for determing the size of the initial design, but it seems intuitive that more is better as it means the sample space will be better saturated. 
initial_design = 500 #Another option that seems to work well: number_of_hypers_to_tune*10 

#Second, generate the design. 
des = lhs(number_of_hypers_to_tune,initial_design,"maximin")

# By default, the lhs samples are unifomrly distributed over the space. We don't really want that all the time though. For example, if we sampled unifomrly here, we might end up with lots of models with nearly 50,000 trees even if we don't strongly belief that 50,000 trees is likely to be optimal. SO, we can skew the samples with some  post-processing. 
# Skewing the number of trees so that we're Sampling more from around the 10k+ range. 
#This is using a beta distribution.  These are great because the distribution is defined on a scale from 0 to 1. So it just shift everything with the existing range. 
des[:,0] = beta.ppf(des[:,0],2,8)
#%%
#You could use this codeblock to visualize the beta distribution being used above.
# import seaborn as sns
#import numpy as np
#from scipy.stats import beta
# Generate some sample data
#data = np.random.uniform(size=1000)
#data = beta.ppf(data,2,8)
# Create a density plot
# sns.kdeplot(data, fill=True, color='blue')
# sns.despine(left=True)
# plt.xlabel('Value')
# plt.ylabel('Density')
# plt.title('Density Plot')
# plt.show()
#%%
#Third, re-scale the design to the intended range and convert elements to integers if needed. 
# The results of lhs are uniformly distributed in the range of 0 to 1.  So, we need to re-scale these values to our desired ranges. To do that we multiply the values by the range and then add the minimum value to all sampled points.  Doing so has the effect of streteching or shrinking the range of the results from 0-1 to whatever we desire and then linearly translating the values.

# Transform the samples to the desired ranges space
des_transformed = []
#Read as, 'for each column'
for i, col in enumerate(des.T):
    #grab the associated minimum and maximums
    hyperparam = list(xgb_params.keys())[i]
    lower, upper, is_int = xgb_params[hyperparam]
    #If is_int is true, then convert the values to an integer.
    if is_int:
        transformed_col = [int(lower + x * (upper - lower)) for x in col]
    #Otherwise, simply rescale.
    else: 
        transformed_col = [lower + x * (upper - lower) for x in col]
    #Save each column to a new object 'des_transformed'
    des_transformed.append(transformed_col)
des_transformed = list(zip(*des_transformed))

#%%

#Step 6, compute the cross-validation estimates for each of the design points. 

#With the initial design now defined, we need to compute correspondign model performance scores.  We can do so with a loop. However, we're going to get a little more fancy than a basic loop here.  Since some models will take longer to fit than others, we don't want there to be any times when a processor is just waiting around for the next instructions while the other models finish. So, we'll use an asyncronous process here.


# First, Define our function that trains the model with each set of hyperparameters on one fold of data
def process_hyperparams(args):
    #'hyperparams' will be the rows in our des_transformed object' 
    hyperparams, (train_index, test_index) = args
    try:
        # Create a dictionary from the hyperparameters
        # We will use the names in xgb_params and assign the values to hyperparams to them.
        hyperparams_dict = {key: value for key, value in zip(xgb_params.keys(), hyperparams)}
        class XGBEstimator:
            def __init__(self, **kwargs):
                self.params = kwargs
            def fit(self, X, y):
                 dtrain = xgb.DMatrix(X, label=y)
                 #self.params.pop() will pull the num_boost_round out of the dictionary we just created above. it needs to be specified like this in python. In R that may be handled internally by other functions. 
                 self.model = xgb.train(params=self.params, dtrain=dtrain, num_boost_round = self.params.pop('num_boost_round'))
                 return self
            def predict(self, X):
                dtest = xgb.DMatrix(X)
                return self.model.predict(dtest)
        pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('model', XGBEstimator(**hyperparams_dict)) #uses the dictionary we defined above.
        ])
        # Split the data into training and validation sets according to our Kfold assignment object (kf)
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        # Train the model and calculate the test error
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        mse = np.mean((y_pred - y_test)**2) #Using a simple means squared error evaluation funciton here. 
        hyperparams_dict['mse'] = mse  #adding a new column with values tot he dictionary. 
        return hyperparams_dict #Returning the dictionary.  
    except Exception as e:   #just in case there are errors this will be thrown and hopefully let you know what went wrong. 
        print(f"Error encountered: {e}")
        raise

# Second, Asynchronously process all hyperparameters and data folds, 2 parts
#Part 1, define the function
def main(X, y, kf, xgb_params, des_transformed):
    with concurrent.futures.ProcessPoolExecutor() as executor:
        # Create a list of tuples where each tuple is (hyperparams, (train_index, test_index))
        inputs = [(hyperparams, split) for hyperparams in des_transformed for split in kf.split(X, y)]
        results = list(executor.map(process_hyperparams, inputs))
        return results

#%%
#Part 2, run the function
import time

if __name__ == '__main__':
    import multiprocessing as mp
    #This is our first multiprocessing spawn point.  everything above this line will get duplicated in memory. not an issue right now as we haven't gotten to GPU work yet.  If you really are limited on RAM, then below you can see how to save the script to a a separate file and call it using subprocess.  That will enable you to explicilty control what gets duplicated. 
    mp.set_start_method('spawn', True)
    start_time = time.time() 

    try:
        results = main(X, y, kf, xgb_params, des_transformed)  #Pass the parameters we need to the main function we defined above. 
    except Exception as e:
        print("An error occurred:", e)
        raise

    print(results)

    end_time = time.time()  # capture end time

    print(f"Total runtime: {end_time - start_time} seconds") #for 500 combinations my system (Ryzen 9 5950x, 128GB ram, approx 90 minutes)

    #Conver the results to a pandas dataframe 
    tested_hypers = pd.DataFrame(results) #This is just a more format similar to an R dataframe. has column names integrated. 
#Save your design matrix because you don't want to have to regenerate it. 
#tested_hypers.to_csv('enter a file path here .csv')

# %%

#Step 7, Fit a Guassian Process to your Design matrix. 
#First, import some more modules.  I put this here instead of at the very top because you can start processing from this block if you restart the session. Also, it helps to clarify what parts of the code are using which modules. 
import numpy as np
import torch
import math
import gpytorch
import pandas as pd
from gpytorch.likelihoods import GaussianLikelihood
from botorch.models import SingleTaskGP
from gpytorch.kernels import SpectralMixtureKernel
from gpytorch.mlls import ExactMarginalLogLikelihood
from torch.optim import AdamW
from sklearn.preprocessing import StandardScaler
#%%
#Second, import your design matrix if you don't already have it in your session. 
tested_hypers = pd.read_csv(r"C:\Users\Zachariah\Desktop\Design_Matrix_500LHS.csv")
tested_hypers = tested_hypers.iloc[:,1:14] #If extra index column was saved, drop it.

#The way the asynchonous routine worked above, each cross-validation routine was saved as a set of 10 rows to the tested_hypers object.  So, if you tested 500 points, you now have an object with 5,000 rows.  So, we need to aggregate those.  A common default choice is to just compute the mean of the set.  If we allowed each fold to have a different amount of training instances, then here we might take a weighted average instead.  Another alterntaive to means is to take a sum score. Means are particularly good, however, because they're unbaised and consistent estimators of the population mean.  In this case the population mean would be something like 'the error we would obtain if fitting on the entire population' and 'unbiased and consistent' implies that although our estimates are still just estimates, in the long-run they converge to the true value over repeated experimentation. 

#The next line computes the means. In words, it says "compute the mean of column 'mse'.  But, first group the observations. Do that by joining the data in columns 0 through 11 before defining unique entries." 
tested_hypers = tested_hypers.groupby(list(tested_hypers.columns[0:11]))['mse'].mean()
#Once cross-validation scores are computed, we need to undo the joining caused by the grouping operation. 
tested_hypers = tested_hypers.reset_index() 


#%%
#Third, load the design matrix to the GPU (the '.cuda()' call is pushing objects to the GPU)
#.cuda() will load the object to the GPU.  the default slot is 0. you can run nvidia-smi in a command prompt to see which card is in 0. You can also specify other cards here .cuda('cuda:1')
GP_X = torch.tensor(tested_hypers.iloc[:,:11].values, dtype=torch.float64).cuda()
GP_y = torch.tensor(tested_hypers.iloc[:,-1].values, dtype=torch.float64).cuda()

# Create a StandardScaler object and fit it to the training data
#This is just standardizing the data to have mean 0 sd 1.  This is helpful as many kernels are sensitive to input scale. Example: RBF kernel. 
scaler = StandardScaler()
#The first line use fit_transform. The 'fit' will save the means and standard deviations. The 'transform' will apply them.
#I've commented this next line out because we'll actually be using a minmax rescale to create a unit cube in this case.  The unit cube is better for the BoTorch work we'll be doing below. 
#GP_X = torch.tensor(scaler.fit_transform(GP_X.cpu().numpy())).cuda()
#If we were doing cross-validation here, then we go to the validation data and only use 'scaler.transform' which applies the means and standard devaitions that were saved during 'fit'.


#Same as a above for dependent variable.  
#However, since it's a single variable, we have to use reshape() so scaler doesn't throw an error and squeeze() so that our backpropogation doesn't throw an error. 
#Scaling the dv when it's just 1 dv isn't strictly required but in my experience it has resulted in more stability and slightly better fits.
GP_y = torch.tensor(scaler.fit_transform(GP_y.cpu().numpy().reshape(-1, 1)).squeeze()).cuda()
GP_y = GP_y.unsqueeze(-1) #necessary or else we'll get an error below about the dimensions of the tensor.

# Min-max scaling for GP_X (This is not strictly necessary but if you use the default liklihood for SingletaskGP (coming up next), it comes with a Gammaprior() that can take advantage of this scaling.
min_x = torch.min(GP_X, axis=0)[0]
max_x = torch.max(GP_X, axis=0)[0]
GP_X = (GP_X - min_x) / (max_x - min_x)

# Min-max scaling for GP_y if you needed.
#min_y = torch.min(GP_y)
#max_y = torch.max(GP_y)
#GP_y = (GP_y - min_y) / (max_y - min_y)

#%%
#Fourth, now we  define the Gaussian Process. 

# We're using GPyTorch here.  This is the best, if not among the best, libraries for fitting GPs.  It is both extremely modular and extremely fast. It can scale GPs to extremely large numbers of observations.  In the millions. And to Billions if you use the recent keOps integration (currently requires a special installation on Windows or the use of Linux)
#In this case, we're using an SingleTaskGP which is an exactgp with some extra methods for BoTorch.  We initialize that in the next lines.  

class GPModel(SingleTaskGP):
    
    def __init__(self, train_x, train_y,likelihood):
        super(GPModel, self).__init__(train_x, train_y,likelihood)
        #We need a liklelihood, a mean, and a covariance. defined here. 
        self.likelihood = likelihood
        #We can use different options for the mean. I think Linear makes sense for bayesian optimization as we are hoping to see that different parts of the space will indicate models which are better than others. a zero mean would imply that across the space we have no expectations about any hyperparameter configurations being any good, same with ConstantMean. 
        #self.mean_module = gpytorch.means.ConstantMean().cuda()
        self.mean_module = gpytorch.means.LinearMean(input_size= GP_X.shape[1]).cuda()
        #self.mean_module = gpytorch.means.ZeroMean().cuda()
        # The kernels.  RBFKernel is less flexible thatn Spectral mixture.  Matern is another population choice.
        #ard refers to applying a different lengthscale to different dimensions.  They can all be the same or different.
        #self.covar_module = gpytorch.kernels.RBFKernel(ard_num_dims=train_x.shape[1]).cuda()

        self.covar_module = SpectralMixtureKernel(num_mixtures=10, ard_num_dims=train_x.shape[1]).cuda()
        #When using spectralmixturekernel, it's smart to initialize the kernel from data.  This ensures that before the model it optimized, the starting conditions are likely to be good.  Bad initialization may result in a poor model fit. We're fitting using the AdamW optimizer. 
        self.covar_module.initialize_from_data(train_x, train_y)
    #We need a forward method.  Data passes 'forward' through the model' 
    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
#We need to define a likelihood. 
#the .double().  By default, the model is formatted with float32.  But, BoTorch prefers float64.  So .double, will make an object be cast in float64. You don't have to use float64 but it gives you more stabilitiy. 
likelihood = GaussianLikelihood().cuda().double()
#If you want to use the prior, here is how. This is putting a prior on the noise, which, in turn, is affecting our kernel.  A prior with smaller numbers here (such as Gammaprior(.5,.005)) is saying that there is a stronger prior belief that the noise is small/low relative to a Gammaprior with higher numbers.
#likelihood = GaussianLikelihood(noise_prior= gpytorch.priors.GammaPrior(1.1,.05)).cuda().double()
#You can also set an initial level, the amount of noise is being estimated.  So this can help you initialize the optimization. Not strictly needed:
#likelihood.noise = torch.tensor([10.0]).cuda()  # Initial noise level 
#Finally, put the pieces together into a instance of the class we defined. 
model = GPModel(GP_X, GP_y, likelihood).cuda().double()

model.train()
likelihood.train()

#%#
#Fifth, define the optimzer. We're using AdamW.  This variant has better properties than the original.
optimizer = AdamW([
    {'params': model.parameters()},  
], lr=0.1)

#Evaluation function
mll = ExactMarginalLogLikelihood(likelihood= likelihood, model= model)

# Create a scheduler for learning rate adjustment
#The next line creates a condition for reducing the learning rate if the best model has n ot been improved upon after 100 iterations during optimization. it sets a new learnign rate of lr*factor (e.g., .1*.5=.05). The idea here is that the program may be jumping over the region of lower loss and that by reducing the learning rate the model will take smaller steps that will hopefully be able to move out of a space where it is stuck. 

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=100, factor=0.25, verbose=True)

n_iterations = 1500
patience = 200
best_loss = float('inf')
best_model_state = None
no_improve_counter = 0

for i in range(n_iterations):
    optimizer.zero_grad()
    output = model(GP_X)
    loss = -mll(output, GP_y)
    loss = loss.mean()
    loss_value = loss.item()

    # Check if loss is NaN and if so, stop the loop
    if math.isnan(loss_value):
        print('Stopping early as loss became NaN at iteration {}'.format(i + 1))
        break

    loss.backward()
    print('Iter %d/%d - Loss: %.3f' % (i + 1, n_iterations, loss_value))

    #The condition below creates an early stopping criteria. No reason to iterate for 5,000 steps if the best model came at 300 steps and by 500 steps there was no improvement still.
    # Update the best loss and best model state if loss is not NaN
    if loss_value < best_loss:
        best_loss = loss_value
        best_model_state = model.state_dict()
        no_improve_counter = 0
    else:
        no_improve_counter += 1

    # Step the scheduler
    scheduler.step(loss)

    if no_improve_counter >= patience:
        print('No improvement for {} iterations, stopping early.'.format(patience))
        break

    optimizer.step()

#%%
#Remove objects no longer needed.
del [output,no_improve_counter,best_loss,loss_value,loss,n_iterations,patience,optimizer,min_x,max_x,scheduler,mll,]
#%%
#Here is how you can reclaim some GPU memory. 
# GP_X = GP_X.to('cpu')
# GP_y = GP_y.to('cpu')
# model = model.to('cpu')
# likelihood = likelihood.to('cpu')
# import gc
# gc.collect()
# torch.cuda.empty_cache()

#%%
#The model is now fit.  Above, we saved the best model during optimization.  Here, we will load that model to the GPU.  Then we will set the model to evaluation mode. By doing so, we're only passing the data 'forward' through the model. 
device = torch.device('cuda')  #ensure model is loaded to GPU. you can still use the .cuda() method.
model.load_state_dict(best_model_state)
#Set the model to evaluation model.  No more need for backpropagation. 
model.eval()



#%%
#This section is how you can make predictions.  In this case, we can see how well the model if fitting to the training data.  This will not be particualrly useful for the present application because the spectral mixture kernel is going to overfit to these data. Generally you want to do cross-validation if you want to have an approximately unbaised estimated of how your model generalizes. 
# GP_X = GP_X.to(device)  # Ensure is on the same device as the model

# likelihood = likelihood.to(device)
# likelihood.eval()

# posterior = model.posterior(GP_X) #We use this version for SingleTaskGP. 
# pred_means = posterior.mean
# pred_variances = posterior.variance
# #In a GPyTorch ExactGP we'd just compute the mean.
# # with torch.no_grad(), gpytorch.settings.fast_pred_var():
# #     observed_pred = likelihood(model(X_train))

# # pred_means = observed_pred.mean
# GP_y = GP_y.to(device)

# SSR = torch.mean((GP_y - pred_means) ** 2)
# SST = torch.mean((GP_y - torch.mean(GP_y)) ** 2)
# R_squared = 1 - SSR / SST

# print('R-squared:', R_squared.item())

# GP_X = GP_X.to('cpu')
# GP_y = GP_y.to('cpu')
# model = model.to('cpu')
# pred_means = pred_means.to('cpu')
# pred_variances = pred_variances.to('cpu')
# del [posterior, SST, SSR, R_squared, pred_means, pred_variances]
# gc.collect()
# torch.cuda.empty_cache()

#%%
#For completeness, this next section goes through a cross-validation routine.  
# from sklearn.model_selection import KFold
# from sklearn.preprocessing import StandardScaler
# import numpy as np

# #The next three lines will initialize model evaluation objects used later below.
# mse_values = []
# preds = []
# y_mean = data[:,-1].mean()

# #If you don't have any prior knowledge of the optimization problem, then it's is a good idea to cross-validate your gaussian process so that you can get a sense of it's appropriateness for your problem. 

# kfold = KFold(n_splits=5, shuffle=True)
 
# for fold, (train_idx, test_idx) in enumerate(kfold.split(data)):
#     print(f'Fold {fold+1}')
#     train_data, test_data = data[train_idx], data[test_idx]
#     X_train, y_train = train_data[:,:-1], train_data[:,-1]
#     X_test, y_test = test_data[:,:-1], test_data[:,-1]

#     # Create a StandardScaler object and fit it to the training data
#     #Again, this is just standardizing the data to have mean 0 sd 1.  This is helpful as many kernels are sensitive to input scale. Example: RBF kernel. 
#     scaler = StandardScaler()
#     #The first line use fit_transform. The 'fit' will save the means and standard deviations. The 'transform' will apply them.
#     X_train = torch.tensor(scaler.fit_transform(X_train.cpu().numpy())).cuda()
#     #Then we go to the validation data and only use 'transform' which applies the means and standard devaitions that were saved during 'fit'.
#     X_test = torch.tensor(scaler.transform(X_test.cpu().numpy())).cuda()
#     #Same as a above for dependent variable.  
#     #However, since it's a single variable, we have to use reshape() so scaler doesn't throw an error and squeeze() so that our backpropogation doesn't throw an error. 
#     #Scaling the dv when it's just 1 dv isn't strictly required but in my experience it has resulted in more stability and slightly better fits.
#     y_train = torch.tensor(scaler.fit_transform(y_train.cpu().numpy().reshape(-1, 1)).squeeze()).cuda()
#     y_test = torch.tensor(scaler.transform(y_test.cpu().numpy().reshape(-1, 1)).squeeze()).cuda()

#     #Now we start defining the Gaussian Process. 
#     class GPModel(ExactGP):
#         def __init__(self, train_x, train_y, likelihood):
#             super(GPModel, self).__init__(train_x, train_y, likelihood)
#             self.mean_module = gpytorch.means.LinearMean(input_size= X_train.shape[1]).cuda()
          
#             self.covar_module = SpectralMixtureKernel(num_mixtures=20, ard_num_dims=train_x.shape[1]).cuda()
#             self.covar_module.initialize_from_data(train_x, train_y)
#             # We can use the LinearKernel if we want to compare our fancy kernel to a simpler one. This is a method that helps disentangle the effects of the Linearmean from above from the spectral mixture kernel.  Basically, if performance is the same when using LinearKernel and LinearMean as when using SpectralMixtureKernel with LinearMean, then the SpectralMixtureKernel hasn't learned anything useful. As the design matrix grows through the optimization problem, that might change though. So, when all is said and done in terms of optimization, it wouldn't hurt for you to evaluate just how well your final surrogate model performs on the problem.  If it fits poorly then you will know that there's a chance you could improve your XGBoost model with further hyperparatmer tuning. 
#             # self.covar_module = gpytorch.kernels.LinearKernel()

#         def forward(self, x):
#             mean_x = self.mean_module(x)
#             covar_x = self.covar_module(x)
#             return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

#     likelihood = GaussianLikelihood().cuda()
#     #Adding noise can help regularize in some cases. 
#     likelihood.noise = torch.tensor([100.0]).cuda()  # Initial noise level

#     model = GPModel(X_train, y_train, likelihood).cuda()

#     model.train()
#     likelihood.train()

#     optimizer = AdamW([
#         {'params': model.parameters()},  
#     ], lr=0.1)

#     mll = ExactMarginalLogLikelihood(likelihood, model)

# scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=100, factor=0.5, verbose=True)

# n_iterations = 5000
# patience = 200 #if we hit 200 iterations with no improvement, then stop. 
# best_loss = float('inf')
# best_model_state = None #tracking the best model
# no_improve_counter = 0

# for i in range(n_iterations):
#     optimizer.zero_grad()
#     output = model(X)
#     loss = -mll(output, y)
#     loss.backward()
#     print('Iter %d/%d - Loss: %.3f' % (i + 1, n_iterations, loss.item()))
#     if loss.item() < best_loss:
#         best_loss = loss.item()
#         best_model_state = model.state_dict()
#         no_improve_counter = 0
#     else:
#         no_improve_counter += 1

#     # Step the scheduler
#     scheduler.step(loss)

#     if no_improve_counter >= patience:
#         print('No improvement for {} iterations, stopping early.'.format(patience))
#         break

#     optimizer.step()

#     print(loss.item()) #Printing the final loss in each fold.

#     device = torch.device('cuda') #making sure model is loaded to GPU
#     model = model.to(device)
#     model.load_state_dict(best_model_state)
#     model.eval()
#     likelihood = likelihood.to(device)
#     likelihood.eval()
#     # Load the best model state
#     model.load_state_dict(best_model_state)
#     # Switch to eval mode
#     model.eval()
#     likelihood.eval()

#     # Compute performance on the test set
#     with torch.no_grad(), gpytorch.settings.fast_pred_var():
#         observed_pred = likelihood(model(X_test))

#     pred_means = observed_pred.mean
#     mse_test = torch.mean((y_test - pred_means) ** 2)
#     mse_values.append(mse_test.item())
#     preds.append(pred_means)

# mean_mse = np.mean(mse_values)
# #If you don't standardize you dependent variable
# mse_y_mean = torch.mean((data[:,-1] - y_mean) ** 2).item()

# print('Mean MSE across all folds:', mean_mse)
# print('MSE of y (i.e., the variance of y):', mse_y_mean)
# #If you don't standardize your depdnent variable 
# #print('Ratio:', 1- mean_mse / mse_y_mean)
# print('Ratio:', 1- mean_mse / 1)

#%%
#Often, people want to make plots of their models predictions to see how they line up with the data.  Here is an example of how to do so assuming you only have 1 dimension to work with. 
# import matplotlib.pyplot as plt

# with torch.no_grad():
#     # Initialize plot
#     f, ax = plt.subplots(1, 1, figsize=(4, 3))

#     # Get upper and lower confidence bounds
#     lower, upper = observed_pred.confidence_region()
    
#     # Plot training data as black stars
#     ax.plot(train_x.numpy(), train_y.numpy(), 'k*')
    
#     # Plot predictive means as blue line
#     ax.plot(test_x.numpy(), observed_pred.mean.numpy(), 'b')
    
#     # Shade between the lower and upper confidence bounds
#     ax.fill_between(test_x.numpy(), lower.numpy(), upper.numpy(), alpha=0.5)
    
#     ax.set_ylim([-3, 3])
#     ax.legend(['Observed Data', 'Mean', 'Confidence'])
#     plt.show()
#%%

# A quick recap. 
#Up to this point we have created an initial design for our Gaussian Process and Fit our Guassian Process.  The next step is to construct our Optimizer to search the Guassian Process for the next candidate to evaluate. There are a number of really good options to choose from today.  In this script we'll use the BoTorch library and we'll implment a one-shot knowledge gradient.  In short, this a relatively new acquisition function that is able to take advantage of of gradient information by using a method which considers the outcome of several possible choices. Since we have gradients available in our gaussian process, using them effectively guides the optimization in an efficient manner. In doing so, it represent the amount of 'knowledge' gained and the uncertainty associated with it when considering candidate points. Doing so is relatively compute intensive. So, if you don't have a decent GPU on hand, you might not use this option. Another option would be CMA-ES.  CMA-ES is a evolutionary strategy which can find the global minimum of a function even if there are hundreds of local minima,  the dimensionality is in the hundreds of dimensions, and the problem is extremely ill-conditioned. It is also derivative free, which means it is broadly applicable to an extremely wide variety of problems. However, CMA-ES needs an evaluation criteria. In this case it would normally be expected improvement. That said, I'm unaware of any empirical work comparing the two appraoches.  

#%%
#Step 8, Implment our Bayesian Optimization routine
import subprocess
from botorch.acquisition import qKnowledgeGradient
from botorch.optim import optimize_acqf

#%%
#First, we need to deinf the bounds for the optimization routine. 

#Start with the dictionary that was previously created. 
xgb_bounds = {
    'num_boost_round': [1,50000,True],
    'max_depth': [1,30,True],
    'learning_rate': [2**-13,2**-1.7,False],
    'min_child_weight': [0,20,False],
    'subsample': [.01,1,False],
    'colsample_bytree': [.01,1,False],
    'colsample_bynode': [.01,1,False],
    'colsample_bylevel': [.01,1,False],
    'alpha': [2**-10,2**7.5, False],
    'lambda': [2**-10,2**7.5,False],
    'gamma': [2**-10,7,False]
}
#Saving the xgb_params so that the worker.py script can load it. 
with open('xgb_params.pkl', 'wb') as f:
    pickle.dump(xgb_bounds, f)

# Remove the boolean entries
xgb_bounds = {k: v[:2] for k, v in xgb_bounds.items()}

# Create a tensor for the lower bounds and upper bounds
lower_bounds = torch.tensor([v[0] for v in xgb_bounds.values()])
upper_bounds = torch.tensor([v[1] for v in xgb_bounds.values()])

# Combine into a 2D tensor
bounds = torch.stack([lower_bounds, upper_bounds]).cuda().double()

#The above lines for 'bounds' is equivalent to this these lines here: 
# bounds = torch.tensor([
#     [1, 1, 2**-13, 0, .01, .01, .01, .01, 2**-10, 2**-10, 2**-10],  # Lower bounds
#     [50000, 25, 2**-1.7, 20, 1, 1, 1, 1, 2**7.5, 2**7.5, 7]  # Upper bounds
# ])


#%%
#Second, Define the number of steps our bayesian optimization routine will take. 
num_iterations = 100
# Third, Initialize the acquisition function.
#We're using one-shot knoweldge gradient. 
#More num_fantasies is better. but at longer runtime and higher memory use. model here is the GP model we fitted above. 

acq_function = qKnowledgeGradient(model=model, num_fantasies=20)

#Fourth, run the routine.
for J in range(num_iterations):

    # Select candidate points. 
    candidates, _ = optimize_acqf(
        acq_function=acq_function,
        bounds=bounds,
        q=3,  # Number of candidates to select a Ryzen 9 5950x has 16 cores (32 threads. So 3x10 =10 models at a time.)
        num_restarts=10,  # Number of random restarts
        raw_samples=128,  # Number of random samples
        options={"batch_limit": 10, "maxiter": 200}, #These control how many points were used in each candidate selection and how many are evalauted at a time. 
        sequential=False #if not using one-shot knowledge gradient, you can choose to add points sequentially. 
    )

    # Evaluate the candidates in the black-box function
   
    new_xs = candidates.detach()
    new_xs = new_xs.to('cpu') 
    new_xs = new_xs.numpy()
    #We'll save these as 'des_transformed' because that's the name of the original variable used in the multiprocessing when we computed estimates for each of the latin-hypercube samples.
    np.save("des_transformed.npy", new_xs)
    #This launches our multiprocessing XGBoost cross-validation routine. That script will import the objects we have saved up to this point in the script and then build each of the xgboost models we need.
    subprocess.run(["python", "worker.py"])
    #The subprocess script will save an obejct we have called 'results' to a file 'results.csv'.
    new_results = pd.read_csv("results.csv",index_col=False)
    #We aggregate the Cross validation estimates as before. 
    new_results = new_results.groupby(list(new_results.columns[0:11]))['mse'].mean()
    #Once cross-validation scores are computed, we need to undo the joining caused by the grouping operation. 
    new_results = new_results.reset_index()
    #append the new results to the set of tested results.
    tested_hypers = pd.concat([tested_hypers,new_results],ignore_index=True)

    #Split into the X and Y again. load on to GPU again. 
    del [GP_X, GP_y] #I put this here just to make sure the GPU ram is released. It's supposed to automatically garbage collect but I'm not sure that happens correctly all the time. 
    GP_X = torch.tensor(tested_hypers.iloc[:,:11].values, dtype=torch.float64).cuda()
    GP_y = torch.tensor(tested_hypers.iloc[:,-1].values, dtype=torch.float64).cuda()
    #scale the Y and min-max rescale the Xs
    scaler = StandardScaler()
    GP_y = torch.tensor(scaler.fit_transform(GP_y.cpu().numpy().reshape(-1, 1)).squeeze(),dtype=torch.float64).cuda()
   
    min_x = torch.min(GP_X, axis=0)[0]
    max_x = torch.max(GP_X, axis=0)[0]
    GP_X = (GP_X - min_x) / (max_x - min_x)
    #Formatting Y. The function expects a 2D object even though its a 1D.
    GP_y = GP_y.unsqueeze(-1)

    # Fit a new model with the updated data
    #del model #again, making sure we keep GPU Ram under control. 
    model = GPModel(GP_X, GP_y, likelihood).cuda().double()
    
    # Set up optimizer and loss function for training
    optimizer = AdamW([{'params': model.parameters()}], lr=0.1)
    mll = ExactMarginalLogLikelihood(likelihood, model)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=100, factor=0.25, verbose=False)

    # Train the model with early stopping, as the original procedure
    n_iterations = 1500
    patience = 200
    best_loss = float('inf')
    best_model_state = None
    no_improve_counter = 0

    for i in range(n_iterations):
        optimizer.zero_grad()
        output = model(GP_X)
        loss = -mll(output, GP_y)
        loss = loss.mean()
        loss_value = loss.item()

     # Check if loss is NaN and if so, stop the loop
        if math.isnan(loss_value):
            print('Stopping early as loss became NaN at iteration {}'.format(i + 1))
            break

        loss.backward()
        #print('Iter %d/%d - Loss: %.3f' % (i + 1, n_iterations, loss_value))

        #The condition below creates an early stopping criteria. No reason to iterate for 5,000 steps if the best model came at 300 steps and by 500 steps there was no improvement still.
        # Update the best loss and best model state if loss is not NaN
        if loss_value < best_loss:
            best_loss = loss_value
            best_model_state = model.state_dict()
            no_improve_counter = 0
        else:
            no_improve_counter += 1

        # Step the scheduler
        scheduler.step(loss)

        if no_improve_counter >= patience:
            #print('No improvement for {} iterations, stopping early.'.format(patience))
            break

        optimizer.step()

     # Load the best state into the model
    model.load_state_dict(best_model_state)
    
    print("MSE for most recent:", [f"{row['mse']:.4f}" for idx, row in tested_hypers.tail(3).iterrows()])
    print("Best Current Score:",[f"{tested_hypers['mse'].min():.4f}"])
    # Update the acquisition function 
    #More on one-shot knowledge gradient.  The idea here is that we're working with a Bayesian model. SO, instead of retraining from scratch, we can just update our exising model. Given that we can do so, one option is to engage in hypotheticals. That is, we can pretend (i.e., fantasize) that we have observed some point and found it's model performance. If we had actually observed the point, we would add that point to our training data and build our model with it.  But, because this is a Bayesian model, we don't need to retrain.  So, instead, we select our pretend point and we update our model with this pretend point.  How does that help us?  Well, we can pick a bunch of different pretend points and estiamte how each will affect our model. Once we have our hypotheticla models, we can measure the difference between our real model and these hypothetical models to estimate which points are providing us with the most real knowledge. While the details are more compelx than this, the key idea is that there should be a correlation between the knowledge gained by "the hypothetical model relative to the current model" and "an truly updated model relative to the current model." So, we can use that informaiton to then select real candidates for testing. In other words, we're trying to take a shortcut by using hypothetical futures to guide our current steps.  
    acq_function = qKnowledgeGradient(model=model, num_fantasies=20)
    #Every 10 iterations we'll save the current results.
    if sum([J+1]) % 10 == 0:
        # save the dataframe as a CSV file every 10 iterations
        filename = f"tested_hypers_iteration_{sum([J+1])}.csv"
        tested_hypers.to_csv("tested_hypers_current.csv", index=False)
