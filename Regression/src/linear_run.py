from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from typing import List

def split(df,features,y,test_size=0.20,random_state=1):
    x_train, x_test , y_train, y_test = train_test_split(df[features],df[y]
    ,test_size=test_size
    ,random_state=random_state) 
    return x_train, x_test, y_train, y_test

### Add Feature Selection
def cleaner():
    """ Input: df, features, target
        Output: X, y for regression. Ready for split
    """
    pass

class executor:
    """
    Df must be clean
    linear only model
    """
    def __init__(self,x_train,x_test,y_train,y_test,model):
        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train
        self.y_test = y_test
        self.model = model


    def scale_fit(self):
        # What is math behind Scaler? Is training per variable? Yes? What can we do about dependent random variables?
        scaler = StandardScaler() 
        self.x_train = scaler.fit_transform(self.x_train) 
        self.x_test = scaler.transform(self.x_test) 

        linear_model = LinearRegression() 
        linear_model.fit(self.x_train,self.y_train)
        return linear_model

        # Model Evaluation
    def eval(self,model):

        y_pred = pd.DataFrame(model.predict(self.x_test))
        pred_train = model.predict(self.x_train)
        print("Train RMSE: %.2f" % mean_squared_error(self.y_train, pred_train,squared=False))
        print("Test RMSE: %.2f" % mean_squared_error(self.y_test, y_pred,squared=False))
        print("Train Coefficient of determination: %.2f" % r2_score(self.y_train, pred_train))
        print("Test Coefficient of determination: %.2f" % r2_score(self.y_test, y_pred))

if __name__ == "__main__":
    import os
    import pandas as pd

    # Load Data - Clean full

    # Load one directory back
    one_back = os.path.normpath(os.getcwd() + os.sep + os.pardir)
    # Load data
    df = pd.read_csv(os.path.join(one_back,'housing_clean.csv'))
    
    features = ['bedrooms_per_house','houses_available','housing_median_age','median_income']
    x_train, x_test, y_train, y_test = split(df,features,'median_house_value',test_size=0.2,random_state=1)
    ex = executor(x_train,x_test,y_train,y_test,model='linear')
    model = ex.scale_fit()
    ex.eval(model)


