import torch                                          
import torch.nn as nn                                   
import torch.optim as optim                             
import pandas as pd                                    
import math                                             
from sklearn.model_selection import train_test_split    
from torch.utils.data import DataLoader, TensorDataset  
import torch.nn.functional as F                         

import os                                              
import numpy as np                                      

import matplotlib.pyplot as plt                          
from scipy.stats import gaussian_kde    

from sklearn.preprocessing import StandardScaler


## set workspace

mydir = "../demo_result/"

if not os.path.exists(mydir):
    os.makedirs(mydir)

os.chdir(mydir)

#####################
# Parameter Setup
#####################

## If the model fails to converge, please consider tuning the following parameters

## Number of epochs
epochs = 50

# ## Number of batches to divide the training set into
# train_batch_count = 50

## Batch size for the training set
# train_batch_size = round(X_train.shape[0] / train_batch_count)
train_batch_size = 100

## Size of the validation set
test_size = 0.2

## learning rate
learning_rate = 0.0001

####################
# Reading expression profile
#####################

# Example usage
file_path = '../demo_data/exp.csv.gz'  # Replace with your file path

expression_profile_df = pd.read_csv(file_path, index_col=0)
expression_profile_df = expression_profile_df.T

# Remove columns that are all zeros
expression_profile_df = expression_profile_df.loc[:, (expression_profile_df != 0).any()]

## log2(exp + 1)
expression_profile_df = np.log2(expression_profile_df + 1)

# ## Expression profile z-score
# zscore_expression_profile_df = expression_profile_df.apply(lambda x: (x - x.mean()) / x.std(), axis=0)

# Normalization: z-score normalization
scaler = StandardScaler()
expression_profile_df = pd.DataFrame(scaler.fit_transform(expression_profile_df), 
                                      index=expression_profile_df.index, 
                                      columns=expression_profile_df.columns)

#######################
# Reading GSVA score
#######################

file_path = '../demo_result/01.oxidative.stress.score.csv'

oxidative_stress_score_df = pd.read_csv(file_path, index_col=0)
oxidative_stress_score_df = oxidative_stress_score_df.T


####################
# Convert data to tensor format
####################

## Take the intersection of samples and ensure the sample order is consistent
samples = np.intersect1d(expression_profile_df.index.values, oxidative_stress_score_df.index.values)

## Convert pandas DataFrames to torch tensors
expression_profile_df = expression_profile_df.loc[samples,:]
expression_profile = torch.tensor(expression_profile_df.values, dtype=torch.float32, device="cuda")

oxidative_stress_score = torch.tensor(oxidative_stress_score_df.loc[samples, ["O.score", "R.score", "OS.score"]].values,  ## First column O, second column R, third column OS
                                      dtype=torch.float32, device="cuda")


#####################
# Split into training and validation sets
######################

# Split the data into training and validation sets
X_train, X_test, OS_train, OS_test = train_test_split(
   expression_profile, oxidative_stress_score, test_size=test_size, random_state=42
)

# Create DataLoader (no targets needed as we only calculate OS from O and R)
train_dataset = TensorDataset(X_train, OS_train)
train_loader = DataLoader(train_dataset, batch_size = train_batch_size, shuffle=True)

test_dataset = TensorDataset(X_test, OS_test)
test_loader = DataLoader(test_dataset, batch_size = X_test.shape[0], shuffle=False)        


###################
# Define Neural Network
####################

class OS_Network(nn.Module):
    def __init__(self, input_size):
        super().__init__()

        self.encoder = nn.Sequential(
            
            # input_layer
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Dropout(0.5),

            # shared_layer1
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Dropout(0.5),

            # shared_layer2
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.5)            
        )
        
        # Prediction heads
        self.o_head = nn.Linear(64, 1)
        self.r_head = nn.Linear(64, 1)
        
        # OS prediction branch (input: O/R predictions)
        self.os_head = nn.Sequential(
            nn.Linear(1 + 1, 64),  # input: o_pred(1) + r_pred(1)
            nn.ReLU(),            
            nn.Linear(64, 1)
            # nn.Tanh()
        )
    
    def forward(self, x):
        # Independent encoding
        feat = self.encoder(x) 
        # Predict O and R
        o_pred = self.o_head(feat)
        r_pred = self.r_head(feat)
        
        # Predict OS (input: o_pred + r_pred)
        os_input = torch.cat([o_pred, r_pred], dim=1)
        os_pred = self.os_head(os_input)
        
        return o_pred, r_pred, os_pred

###############
# Define loss function
###############

## Used to compute MSE on the test set
mse_test = nn.MSELoss()

## Test the correlation between GSVA and our scores
def pearson_correlation(x, y):
    """
    Calculate the Pearson correlation coefficient between two tensors.
    """
    vx = x - torch.mean(x)
    vy = y - torch.mean(y)
    correlation = torch.sum(vx * vy) / (torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2)) + 0.001)
    return correlation



class CustomLoss(nn.Module):
    def __init__(self, lambda_r=0.5, lambda_o=0.5):
        super().__init__()
        self.mse = nn.MSELoss()
        self.lambda_r = lambda_r
        self.lambda_o = lambda_o
        
    def forward(self, o_pred, r_pred, os_pred, batch_os):
        # Base loss
        loss_o = self.mse(o_pred.squeeze(), batch_os[:,0])
        loss_r = self.mse(r_pred.squeeze(), batch_os[:,1])
        loss_os = self.mse(os_pred.squeeze(), batch_os[:,2])
        
        # Constraint 1: R negatively regulates OS (d(OS)/d(R) < 0)
        r_grad = torch.autograd.grad(
            outputs=os_pred, 
            inputs=r_pred, 
            grad_outputs=torch.ones_like(os_pred),
            retain_graph=True,
            create_graph=True
        )[0]
        r_constraint = torch.mean(F.relu(r_grad))  # Correct direction
        
        # Constraint 2: O positively regulates OS (d(OS)/d(O) > 0)
        o_grad = torch.autograd.grad(
            outputs=os_pred, 
            inputs=o_pred, 
            grad_outputs=torch.ones_like(os_pred),
            retain_graph=True,
            create_graph=True
        )[0]
        o_constraint = torch.mean(F.relu(-o_grad))  # Correct direction
        
        total_loss = (
            loss_o + loss_r + loss_os +
            self.lambda_r * r_constraint +
            self.lambda_o * o_constraint
        )
        return total_loss


###################
# Instantiate model and optimizer
###################

## Initialize model
model = OS_Network(input_size=X_train.shape[1])
model = model.to('cuda')

## Initialize loss function
criterion = CustomLoss()

# Define optimizer
optimizer = optim.Adam(model.parameters(), lr = learning_rate, weight_decay=1e-4)


###############
# Training loop
################

## Set model to training mode
model.train()

# Store loss for each iteration
loss_train = []

for epoch in range(epochs):
    
    running_loss = 0.0

    for inputs, batch_os in train_loader:
        
       # Update gradients
        optimizer.zero_grad()
        
        # Forward pass
        o_pred, r_pred, os_pred = model(inputs)
        
        # Compute loss
        # loss = criterion(outputs, targets) 
        loss = criterion(o_pred, r_pred, os_pred, batch_os) 
        loss_train.append(loss.item())
        running_loss += loss.item()
        
        # Backward pass and optimization
        loss.backward()
        optimizer.step()


    # loss_train.append(running_loss / len(train_loader))  ## Average loss for one epoch

    if (epoch + 1) % 1 == 0:
        print(f'Epoch [{epoch+1}/{epochs}], Mean Loss: {(running_loss / len(train_loader)):.4f}')


        ## Output loss
        df = pd.DataFrame({
            'Iteration': range(len(loss_train)),
            'loss': loss_train
        })
        
        df.to_csv(mydir + '/01.train.loss.csv', index=False)

#################
# Plot training loss
#################

plt.figure()                                                                ## Create a new figure object
line, = plt.plot(list(range(len(loss_train))), loss_train)
plt.legend([line], ["Loss value"])
plt.ylabel("Loss")                                                          ## Set the label for the y-axis
plt.xlabel("Iteration")                                                     ## Set the label for the x-axis

plt.savefig(mydir + '/01.train.loss.pdf', format='pdf')                     ## Save as a PDF file
plt.show()                                                                  ## Never use show() before plt.savefig. The figure will not be displayed in savefig after saving
plt.close()  # Close the canvas to prevent redundant plotting


##########################
# Evaluation loop (for test set)
##########################

## Set model to evaluation mode (test set)
model.eval()

loss_test = []
loss_o_all = []
loss_r_all = []
loss_os_all = []
loss_r_constraint_all = []
loss_o_constraint_all = []


# with torch.no_grad():
for inputs, batch_os in test_loader:

    # Update gradients
    optimizer.zero_grad()
    
    inputs = inputs.requires_grad_(True)
    
    # Forward pass
    o_pred, r_pred, os_pred = model(inputs)

    # Compute loss
    # loss = criterion(outputs, targets) 
    loss = criterion(o_pred, r_pred, os_pred, batch_os) 
    
    loss_test.append(loss.item())

    ## Correlation with GSVA scores in the test set
    ## OS calculated based on O and R is correlated with GSVA's OS score
    

    loss_o_all.append(mse_test(o_pred.squeeze(), batch_os[:,0]).to("cpu").detach().numpy())
    loss_r_all.append(mse_test(r_pred.squeeze(), batch_os[:,1]).to("cpu").detach().numpy())
    loss_os_all.append(mse_test(os_pred.squeeze(), batch_os[:,2]).to("cpu").detach().numpy())
    
    # Constraint 1: R negatively regulates OS (d(OS)/d(R) < -0.1)
    r_grad = torch.autograd.grad(
        outputs=os_pred, 
        inputs=r_pred, 
        grad_outputs=torch.ones_like(os_pred),
        retain_graph=True,
        create_graph=True
    )[0]
    loss_r_constraint_all.append(torch.mean(F.relu(r_grad + 0.1)).to("cpu").detach().numpy())  # Correct direction
    
    # Constraint 2: O positively regulates OS (d(OS)/d(O) > 0.1)
    o_grad = torch.autograd.grad(
        outputs=os_pred, 
        inputs=o_pred, 
        grad_outputs=torch.ones_like(os_pred),
        retain_graph=True,
        create_graph=True
    )[0]
    loss_o_constraint_all.append(torch.mean(F.relu(0.1 - o_grad)).to("cpu").detach().numpy())  # Correct direction

    print(f'Test Loss: {loss.item():.4f}')

## Output loss
df = pd.DataFrame({
    'Iteration': range(len(loss_test)),
    'mse_o': loss_o_all,
    'mse_r': loss_r_all,
    'mse_os': loss_os_all,
    'loss_r_constraint': loss_r_constraint_all,
    'loss_o_constraint': loss_o_constraint_all,
    'loss.total': loss_test
})

df.to_csv(mydir + '/02.test.loss.csv', index=False)    

####################
# Save the model
####################

## Set model to evaluation mode (test set)
model.eval()

# Save the entire model
torch.save(model, mydir + '/03.01.model.pth')

# Save model weights
torch.save(model.state_dict(), mydir + '/03.02.model_weights.pth')

####################
# Output oxidative stress scores for all samples
####################

## Set model to evaluation mode (test set)
model.eval()

with torch.no_grad():
    
    o_pred, r_pred, os_pred = model(expression_profile)

df = pd.DataFrame({
    'sample': expression_profile_df.index,    ## expression_profile_df and expression_profile are in the same order
    'o_score': o_pred.to('cpu').numpy().squeeze(),
    'r_score': r_pred.to('cpu').numpy().squeeze(),
    'os_score': os_pred.to('cpu').numpy().squeeze()
})

df.to_csv(mydir + '/04.os.score.csv', index=False)

##################
# Perform 0,1 normalization
##################

# Normalize all columns except for 'sample'
cols_to_normalize = ["o_score", "r_score", "os_score"]
df[cols_to_normalize] = df[cols_to_normalize].apply(lambda x: (x - x.min()) / (x.max() - x.min()))

df.to_csv(mydir + '/05.os.score.scale.csv', index=False)
