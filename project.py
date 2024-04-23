# 
# Author: Danielle N. Cunes
# Date: 4/22/2024
# Course: ISTA 131
# Assignment: Final Project
# File: project.py
# Purpose: The purpose of this program is to 
# 

import pandas as pd, numpy as np
import matplotlib.pyplot as plt

def read_in_csv():
    '''
    This function uses the read_csv function to read in a csv and create a dataframe using the csv file.
    
    Parameters: 
        None

    Returns:
        grad_students_df: a pandas DataFrame
    '''
    column_names = ['Year', 'Total Grad Students', 'Male Grad Students', 'Male Grad Percent', 'Female Grad Students', 'Female Grad Percent', 
                    'Total Post-Doc Students', 'Male Post-Doc Students', 'Male Post-Doc Percent', 'Female Post-Doc Students', 'Female Post-Doc Percent', 
                    'Total Doctorate Students', 'Male Doctorate Students', 'Male Doctorate Percent', 'Female Doctorate Students', 'Female Doctorate Percent']
    
    grad_students_df = pd.read_csv("grad_students.csv", skiprows=[0,1,2,3,4,5,7,30,31,34,35,36,37,38,39,40,41,42,43,44,45,48,49,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69], names=column_names,
                                   thousands=',')

    print(grad_students_df)

    return grad_students_df
    
def clean_up_data(grad_students_df):
    '''
    This function takes the DataFrame created in the previous function, cleans up the data by filling in missing data and extracting the data 
    needed to plot scatterplots, line plots, and bar charts.
    
        Parameters: 
            grad_students_df: a pandas DataFrame

        Returns:
            grad_students_df: a pandas DataFrame
    '''
    grad_students_df = grad_students_df.replace('na', np.nan)
    grad_students_df = grad_students_df.fillna(method="ffill").fillna(method="bfill")
    grad_students_df = grad_students_df.apply(lambda row: row.fillna(row.interpolate()), axis=1)

    return grad_students_df

def female_grad_scatterplot(grad_students_df):
    '''
    This function 
    
        Parameters: 
            

        Returns:
            
    '''
    grad_students_df['Female Grad Students'] = pd.to_numeric(grad_students_df['Female Grad Students'])
    x = grad_students_df['Year']
    y = grad_students_df['Female Grad Students']

    coefficients = np.polyfit(x,y,1)
    poly = np.poly1d(coefficients)

    plt.scatter(x,y)

    plt.plot(x, poly(x), color='magenta')

    plt.xlabel('Year', color='darkmagenta')
    plt.tick_params(axis='x', colors='darkmagenta')
    plt.ylabel('Female Grad Students', color='darkmagenta')
    plt.tick_params(axis='y', colors='darkmagenta')
    plt.gca().spines['top'].set_color('darkmagenta')
    plt.gca().spines['bottom'].set_color('darkmagenta')
    plt.gca().spines['left'].set_color('darkmagenta')
    plt.gca().spines['right'].set_color('darkmagenta')
    plt.gca().set_facecolor('lavenderblush')
    plt.title('Female Grad Students with Regression Line', fontdict={'fontsize': 16, 'fontweight': 'bold', 'color': 'darkmagenta'})
    plt.scatter(grad_students_df['Year'], grad_students_df['Female Grad Students'], label='Female Grad Students', c='hotpink')
    plt.legend()

def create_bar_chart():
    '''
    This function 
    
        Parameters: 
            

        Returns:
            
    '''
    pass

def create_line_plot():
    '''
    This function 
    
        Parameters: 

        Returns:
            
    '''
    pass

def main():
    new_df = read_in_csv()
    print(clean_up_data(new_df))
    female_grad_scatterplot(new_df)
    plt.show()

if __name__ == "__main__":
    main()